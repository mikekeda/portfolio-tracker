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

from backend.screener_config import SCORE_NORMALIZER
from backend.utils.risk_math import ledoit_wolf_cov, hrp_weights, risk_contributions
from backend.views._shared import PRICE_COLUMN
from config import SPY
from models import (
    CurrencyRateDaily,
    FeaturesDaily,
    Instrument,
    InstrumentYahoo,
    PricesDaily,
    SecFeaturesDaily,
)

_GBP_FACTORS = {"GBP": 1.0, "GBX": 0.01, "GBp": 0.01}

# Fundamental snapshot older than this at decision time is treated as missing —
# a stale row from a stopped writer must not silently steer trades for weeks.
FUNDAMENTAL_STALENESS_DAYS = 7
# sec_features_daily is month-end grain; allow a full month of slack for SEC rows only.
SEC_FUNDAMENTAL_STALENESS_DAYS = 45
MIN_HISTORY_DAYS = 252  # a symbol enters the tradable universe after a year of prices
RISK_WINDOW_DAYS = 504  # trailing window for HRP / risk contributions
MIN_RISK_OBS = 300

# Statement Tier B columns present on sec_features_daily (no screener / analyst / DCF).
SEC_FUNDAMENTAL_COLUMNS = (
    "roic",
    "gross_margin",
    "operating_margin",
    "profit_margin",
    "revenue_growth",
    "fcf_yield",
    "debt_to_equity",
    "rule_of_40",
    "f_score",
    "roic_ttm",
    "roic_ttm_trend",
    "gross_margin_trend_4q",
    "operating_margin_trend_4q",
    "revenue_growth_4q_avg",
)

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
    "screener_score_max",
    "roic_ttm",
    "roic_ttm_trend",
    "gross_margin_trend_4q",
    "operating_margin_trend_4q",
    "revenue_growth_4q_avg",
    "eps_revision_ratio_30d",
    "eps_next_q_growth",
)

# screener_score is not comparable across these dates — each changed how it is computed.
SCREENER_CUTOVERS = (
    date(2026, 7, 30),  # currency-correct FCF, fixed BB-width percentile, margin gates, reweighting
    date(2026, 8, 23),  # fcf_margin/fcf_yield moved off Yahoo's levered freeCashflow (utils/fcf.py)
)


@dataclass
class MarketData:
    """Everything the agent needs, loaded once."""

    gbp_prices: pd.DataFrame  # dates × symbols, GBP-normalized adjusted close
    currencies: dict[str, str]  # symbol -> native currency
    tags: dict[str, list[str]]  # symbol -> instrument tags
    etf_symbols: frozenset[str]  # quoteType == 'ETF' (exempt from UK stamp duty)
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
            etf_symbols=self.etf_symbols,
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

    etf_rows = (
        await session.execute(
            select(Instrument.yahoo_symbol)
            .join(InstrumentYahoo, InstrumentYahoo.instrument_id == Instrument.id)
            .where(InstrumentYahoo.info["quoteType"].astext == "ETF")
        )
    ).scalars()
    etf_symbols = frozenset(etf_rows)

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
    yahoo_keys: set[tuple[date, str]] = set()
    for f in feat_rows:
        sym = symbol_by_id.get(f.instrument_id)
        if sym is None:
            continue
        yahoo_keys.add((f.date, sym))
        rec = {"date": f.date, "symbol": sym, "fundamentals_source": "yahoo"}
        for col in FUNDAMENTAL_COLUMNS:
            rec[col] = getattr(f, col)
        # Sector exclusions cap what a stock can score, so rank on the ratio.
        # Pre-cutover rows have no max and fall back to the flat normalizer.
        if rec["screener_score"] is not None:
            rec["screener_score"] = rec["screener_score"] / (rec["screener_score_max"] or SCORE_NORMALIZER)
        eval_ = f.thesis_rule_eval or {}
        # sell_signal conflates "a sell rule fired" with "above allocation band";
        # the strategy exits on the former but only trims to the band on the latter.
        rec["thesis_sell_fired"] = bool(eval_.get("sell_rules_met"))
        rec["thesis_buy_fired"] = bool(eval_.get("buy_rules_met"))
        rec["thesis_above_max"] = eval_.get("allocation_status") == "above_max"
        rec["thesis_target_max"] = eval_.get("target_weight_max_pct")  # percent or None
        rec["thesis_sell_reasons"] = "; ".join(
            r.get("description") or r.get("reason", "") for r in (eval_.get("sell_rules_met") or [])
        )
        records.append(rec)

    # SEC PIT panel fills dates/symbols Yahoo FeaturesDaily does not cover.
    # Statement Tier B only — screener_score stays NaN (never mix SEC into scored
    # screeners without an explicit cutover rebuild). Backfilled-era composite is
    # ~68% Tier A; quality gate collapses to ROIC >= 15; prefer CIK-only universe.
    sec_rows = (
        await session.execute(
            select(SecFeaturesDaily).where(*( [SecFeaturesDaily.date <= end] if end else [] ))
        )
    ).scalars()
    for f in sec_rows:
        sym = symbol_by_id.get(f.instrument_id)
        if sym is None or (f.date, sym) in yahoo_keys:
            continue
        rec = {"date": f.date, "symbol": sym, "fundamentals_source": "sec"}
        for col in FUNDAMENTAL_COLUMNS:
            rec[col] = getattr(f, col, None) if col in SEC_FUNDAMENTAL_COLUMNS else None
        rec["thesis_sell_fired"] = False
        rec["thesis_buy_fired"] = False
        rec["thesis_above_max"] = False
        rec["thesis_target_max"] = None
        rec["thesis_sell_reasons"] = ""
        records.append(rec)

    fundamentals = pd.DataFrame(records)
    if not fundamentals.empty:
        fundamentals = fundamentals.sort_values(["symbol", "date"]).reset_index(drop=True)
        # Consecutive-snapshot streak of a fired sell rule, ending at each row.
        # Cumulative within run-groups → backward-only, so truncating history
        # never changes past streak values (invariance-check safe).
        fundamentals["thesis_sell_streak"] = fundamentals.groupby("symbol")["thesis_sell_fired"].transform(
            lambda s: s.astype(int).groupby((~s).cumsum()).cumsum()
        )

    return MarketData(
        gbp_prices=gbp_prices,
        currencies=currencies,
        tags=tags,
        etf_symbols=etf_symbols,
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


def use_sec_universe(md: MarketData, d: date) -> bool:
    """True when fundamentals on/before d are SEC-only (no Yahoo FeaturesDaily yet).

    Activates the CIK-only tradable filter so non-CIK names cannot free-pass
    `_quality_ok` or warp z-scores in the backfilled era.
    """
    if md.fundamentals.empty or "fundamentals_source" not in md.fundamentals.columns:
        return False
    prior = md.fundamentals[md.fundamentals["date"] <= d]
    if prior.empty:
        return False
    return not (prior["fundamentals_source"] == "yahoo").any() and (prior["fundamentals_source"] == "sec").any()


def tradable_universe(md: MarketData, d: date, *, sec_only: bool = False) -> list[str]:
    """Symbols priced on d with at least a year of prior history.

    When `sec_only` is True (backfilled-era SEC panel), keep symbols that have a
    sec fundamentals row on or before d — CIK coverage only. Mixing non-CIK names
    free-passes `_quality_ok` and warps z-scores.
    """
    if d not in md.gbp_prices.index:
        return []
    priced = md.gbp_prices.loc[d].notna()
    enough_history = md.gbp_prices.loc[:d].notna().sum() >= MIN_HISTORY_DAYS
    symbols = sorted(md.gbp_prices.columns[priced & enough_history])
    if not sec_only or md.fundamentals.empty or "fundamentals_source" not in md.fundamentals.columns:
        return symbols
    prior = md.fundamentals[md.fundamentals["date"] <= d]
    sec_syms = set(prior.loc[prior["fundamentals_source"] == "sec", "symbol"])
    return [s for s in symbols if s in sec_syms]


def features_for_date(
    md: MarketData, d: date, universe: list[str], fundamentals_as_of: date | None = None
) -> pd.DataFrame:
    """One row per symbol: technicals at d + freshest fundamentals.

    Fundamentals are as-of `d` by default (strict point-in-time for backtests).
    The live runner passes `fundamentals_as_of=today`: its decision date is the
    last *trading* day, but a suggestion executed tomorrow may legitimately use
    a feature snapshot taken over the weekend.

    Yahoo rows use FUNDAMENTAL_STALENESS_DAYS (7); SEC month-end rows use
    SEC_FUNDAMENTAL_STALENESS_DAYS (45). Thesis flags on SEC rows are forced
    False (unknown → no EXIT escalation), not NaN.
    """
    cross = pd.DataFrame({name: matrix.loc[d].reindex(universe) for name, matrix in md.technicals.items()})

    if md.fundamentals.empty:
        return cross

    as_of = fundamentals_as_of or d
    fund = md.fundamentals[md.fundamentals["date"] <= as_of]
    if "fundamentals_source" in fund.columns:
        yahoo_cut = as_of - timedelta(days=FUNDAMENTAL_STALENESS_DAYS)
        sec_cut = as_of - timedelta(days=SEC_FUNDAMENTAL_STALENESS_DAYS)
        window = fund[
            ((fund["fundamentals_source"] == "yahoo") & (fund["date"] >= yahoo_cut))
            | ((fund["fundamentals_source"] == "sec") & (fund["date"] >= sec_cut))
            | (~fund["fundamentals_source"].isin(("yahoo", "sec")) & (fund["date"] >= yahoo_cut))
        ]
    else:
        fresh_cutoff = as_of - timedelta(days=FUNDAMENTAL_STALENESS_DAYS)
        window = fund[fund["date"] >= fresh_cutoff]
    assert window.empty or window["date"].max() <= as_of, "fundamental row dated after decision date"
    latest = window.groupby("symbol").tail(1).set_index("symbol").drop(columns=["date"], errors="ignore")
    out = cross.join(latest.reindex(universe))
    # Boolean flags must never be NaN: bool(NaN) is True, and a missing thesis
    # evaluation must read as "no signal", not as a fired sell rule.
    for col in ("thesis_sell_fired", "thesis_buy_fired", "thesis_above_max"):
        if col in out.columns:
            out[col] = out[col].fillna(False).astype(bool)
    if "thesis_sell_reasons" in out.columns:
        out["thesis_sell_reasons"] = out["thesis_sell_reasons"].fillna("")
    if "thesis_sell_streak" in out.columns:
        out["thesis_sell_streak"] = out["thesis_sell_streak"].fillna(0).astype(int)
    return out


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
