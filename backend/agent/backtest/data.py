"""Market data loading and point-in-time feature construction for the agent.

Bulk-loads prices/FX/features once into wide pandas matrices (dates × symbols)
and derives all price-based technicals with backward-looking rolling windows,
which makes them leak-resistant by construction: the row for date d only ever
depends on data at or before d. Used by both the backtest engine and the live
daily runner so the two paths cannot diverge.

Prices are GBP-normalized adjusted closes (total-return series). Fills also use
adjusted closes: PricesDaily.open_price is split-unadjusted, so mixing it with
adjusted closes would corrupt quantities around splits.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.utils.risk_math import ledoit_wolf_cov, hrp_weights, risk_contributions
from backend.views._shared import PRICE_COLUMN
from config import SPY
from models import CurrencyRateDaily, FeaturesDaily, Instrument, PricesDaily

_GBP_FACTORS = {"GBP": 1.0, "GBX": 0.01, "GBp": 0.01}

# Fundamental snapshot older than this at decision time is treated as missing —
# a stale row from a stopped writer must not silently steer trades for weeks.
FUNDAMENTAL_STALENESS_DAYS = 7
MIN_HISTORY_DAYS = 252  # a symbol enters the tradable universe after a year of prices
RISK_WINDOW_DAYS = 504  # trailing window for HRP / risk contributions
MIN_RISK_OBS = 300

TECHNICAL_NAMES = (
    "mom_12_1",
    "mom_6m",
    "rs_6m_vs_spy",
    "dist_sma200",
    "dist_sma50",
    "dist_52w_high",
    "rsi_14",
    "vol_90d",
)

FUNDAMENTAL_COLUMNS = (
    "roic",
    "gross_margin",
    "operating_margin",
    "profit_margin",
    "revenue_growth",
    "fcf_yield",
    "debt_to_equity",
    "short_percent_float",
    "analyst_rec_mean",
    "analyst_target_upside",
    "forward_pe",
    "peg",
    "pe_5y_avg_vs_current_pct",
    "dcf_implied_growth",
    "rule_of_40",
    "f_score",
    "screener_score",
)


@dataclass
class MarketData:
    """Everything the agent needs, loaded once."""

    gbp_prices: pd.DataFrame  # dates × symbols, GBP-normalized adjusted close
    currencies: dict[str, str]  # symbol -> native currency
    tags: dict[str, list[str]]  # symbol -> instrument tags
    technicals: dict[str, pd.DataFrame]  # name -> dates × symbols matrix
    fundamentals: pd.DataFrame  # long frame: date, symbol, FUNDAMENTAL_COLUMNS…; may be empty

    def truncated(self, cutoff: date) -> "MarketData":
        """Copy with every data source hard-cut at `cutoff` (invariance checks).

        Technicals are recomputed from the truncated prices, not sliced from the
        precomputed matrices — otherwise a look-ahead bug inside
        compute_technicals would survive the invariance check undetected.
        """
        prices = self.gbp_prices.loc[:cutoff]
        return MarketData(
            gbp_prices=prices,
            currencies=self.currencies,
            tags=self.tags,
            technicals=compute_technicals(prices),
            fundamentals=self.fundamentals[self.fundamentals["date"] <= cutoff]
            if not self.fundamentals.empty
            else self.fundamentals,
        )


async def load_market_data(session: AsyncSession, start: date, end: date | None = None) -> MarketData:
    """Bulk-load prices, FX, instrument metadata and FeaturesDaily history.

    `start` is the first backtest/decision date; prices load from a year
    earlier so rolling features are warm from day one.
    """
    inst_rows = (
        await session.execute(
            select(Instrument.id, Instrument.yahoo_symbol, Instrument.currency, Instrument.tags).where(
                Instrument.yahoo_symbol.is_not(None)
            )
        )
    ).all()
    currencies = {r.yahoo_symbol: r.currency for r in inst_rows}
    tags = {r.yahoo_symbol: (r.tags or []) for r in inst_rows}
    symbol_by_id = {r.id: r.yahoo_symbol for r in inst_rows}

    load_start = start - timedelta(days=MIN_HISTORY_DAYS * 2)
    price_rows = (
        await session.execute(
            select(PricesDaily.symbol, PricesDaily.date, PRICE_COLUMN).where(
                PricesDaily.symbol.in_(list(currencies)),
                PricesDaily.date >= load_start,
                *( [PricesDaily.date <= end] if end else [] ),
            )
        )
    ).all()
    prices = (
        pd.DataFrame(price_rows, columns=["symbol", "date", "price"])
        .pivot(index="date", columns="symbol", values="price")
        .sort_index()
    )

    fx_rows = (
        await session.execute(
            select(CurrencyRateDaily.date, CurrencyRateDaily.from_currency, CurrencyRateDaily.rate).where(
                CurrencyRateDaily.to_currency == "GBP", CurrencyRateDaily.date >= load_start
            )
        )
    ).all()
    fx = (
        pd.DataFrame(fx_rows, columns=["date", "currency", "rate"])
        .pivot(index="date", columns="currency", values="rate")
        .reindex(prices.index)
        .ffill()
    )

    factors = {}
    for sym in prices.columns:
        ccy = currencies.get(sym)
        if ccy in _GBP_FACTORS:
            factors[sym] = pd.Series(_GBP_FACTORS[ccy], index=prices.index)
        else:
            factors[sym] = fx.get(ccy, pd.Series(np.nan, index=prices.index))
    gbp_prices = prices * pd.DataFrame(factors)

    feat_rows = (
        await session.execute(
            select(FeaturesDaily).where(*( [FeaturesDaily.date <= end] if end else [] ))
        )
    ).scalars()
    records = []
    for f in feat_rows:
        sym = symbol_by_id.get(f.instrument_id)
        if sym is None:
            continue
        rec = {"date": f.date, "symbol": sym}
        for col in FUNDAMENTAL_COLUMNS:
            rec[col] = getattr(f, col)
        eval_ = f.thesis_rule_eval or {}
        rec["thesis_sell_fired"] = bool(eval_.get("sell_signal"))
        rec["thesis_buy_fired"] = bool(eval_.get("buy_signal"))
        records.append(rec)
    fundamentals = pd.DataFrame(records)
    if not fundamentals.empty:
        fundamentals = fundamentals.sort_values(["symbol", "date"]).reset_index(drop=True)

    return MarketData(
        gbp_prices=gbp_prices,
        currencies=currencies,
        tags=tags,
        technicals=compute_technicals(gbp_prices),
        fundamentals=fundamentals,
    )


def compute_technicals(gbp_prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All Tier-A feature matrices, vectorized, backward-looking only."""
    p = gbp_prices
    sma50 = p.rolling(50, min_periods=50).mean()
    sma200 = p.rolling(200, min_periods=200).mean()
    returns = p.pct_change()

    mom_6m = p / p.shift(126) - 1
    tech = {
        "mom_12_1": p.shift(21) / p.shift(252) - 1,
        "mom_6m": mom_6m,
        "dist_sma200": p / sma200 - 1,
        "dist_sma50": p / sma50 - 1,
        "dist_52w_high": p / p.rolling(252, min_periods=252).max() - 1,
        "rsi_14": _wilder_rsi(p, 14),
        "vol_90d": returns.rolling(90, min_periods=60).std() * np.sqrt(252),
    }
    if SPY in p.columns:
        tech["rs_6m_vs_spy"] = mom_6m.sub(mom_6m[SPY], axis=0)
    else:
        tech["rs_6m_vs_spy"] = pd.DataFrame(np.nan, index=p.index, columns=p.columns)
    return tech


def _wilder_rsi(prices: pd.DataFrame, period: int) -> pd.DataFrame:
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def tradable_universe(md: MarketData, d: date) -> list[str]:
    """Symbols priced on d with at least a year of prior history."""
    if d not in md.gbp_prices.index:
        return []
    priced = md.gbp_prices.loc[d].notna()
    enough_history = md.gbp_prices.loc[:d].notna().sum() >= MIN_HISTORY_DAYS
    return sorted(md.gbp_prices.columns[priced & enough_history])


def features_for_date(md: MarketData, d: date, universe: list[str]) -> pd.DataFrame:
    """One row per symbol: technicals at d + freshest fundamentals dated ≤ d."""
    cross = pd.DataFrame({name: matrix.loc[d].reindex(universe) for name, matrix in md.technicals.items()})

    if md.fundamentals.empty:
        return cross

    fresh_cutoff = d - timedelta(days=FUNDAMENTAL_STALENESS_DAYS)
    window = md.fundamentals[(md.fundamentals["date"] <= d) & (md.fundamentals["date"] >= fresh_cutoff)]
    assert window.empty or window["date"].max() <= d, "fundamental row dated after decision date"
    latest = window.groupby("symbol").tail(1).set_index("symbol").drop(columns=["date"])
    return cross.join(latest.reindex(universe))


def risk_columns(md: MarketData, d: date, weights: dict[str, float]) -> pd.DataFrame:
    """HRP weight and fractional risk contribution for the held book at d.

    Symbols without MIN_RISK_OBS trailing observations get NaN (young listings);
    the strategy treats NaN as "no risk tilt available".
    """
    held = [s for s, w in weights.items() if w > 0 and s in md.gbp_prices.columns]
    empty = pd.DataFrame(columns=["hrp_weight", "risk_contribution"], dtype=float)
    if len(held) < 2:
        return empty

    window = md.gbp_prices.loc[:d, held].tail(RISK_WINDOW_DAYS)
    returns = window.pct_change().iloc[1:]
    usable = [s for s in held if returns[s].notna().sum() >= MIN_RISK_OBS]
    if len(usable) < 2:
        return empty
    matrix = returns[usable].dropna()
    if len(matrix) < MIN_RISK_OBS:
        return empty

    cov, _ = ledoit_wolf_cov(matrix.to_numpy())
    hrp = hrp_weights(cov)
    w = np.array([weights[s] for s in usable])
    w = w / w.sum()
    rc, _ = risk_contributions(cov, w)
    return pd.DataFrame({"hrp_weight": hrp, "risk_contribution": rc}, index=usable)
