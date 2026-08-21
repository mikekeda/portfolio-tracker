"""
Risk endpoint — feeds the /risk page with HRP reference weights and
risk-contribution diagnostics.

Builds a GBP daily-returns matrix for current holdings from PricesDaily +
CurrencyRateDaily, estimates a Ledoit-Wolf shrunk covariance, then returns:
  * annualised portfolio volatility, Meucci effective bets, and the
    correlation-blind risk-concentration count alongside it
  * per-holding risk contribution vs value weight, HRP reference weight,
    and the volatility impact of adding NEW_MONEY_GBP to that position
  * the same volatility impact for every non-held monitored instrument
    (closed form against the portfolio return series; see _candidate_vol_deltas)
  * per-tag-cluster value weight vs risk share (against the soft caps), and the
    breached tags blocking each holding from receiving new money
  * the latest stored Sharpe / Sortino / Jensen's alpha / beta alongside

Holdings with too little price history (recent IPOs) are excluded from the
model and reported separately. The payload is cached in-process; it only
changes when new daily prices land.
"""

import time
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import aliased
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agent.constraints import ETF_TAG
from backend.schemas.instrument_thesis import InstrumentThesisSchema
from backend.app import get_db_session
from backend.utils.risk_math import (
    correlation_clusters,
    effective_bets,
    hrp_weights,
    ledoit_wolf_cov,
    risk_contributions,
)
from backend.views._shared import PRICE_COLUMN
from config import (
    LOOKTHROUGH_MIN_EXPOSURE_GBP,
    MAX_TOLERABLE_ANNUAL_LOSS,
    logger,
    SPY,
    TAG_CLUSTER_SOFT_CAPS,
    TAG_RISK_ALERT_PCT,
    TRADING_DAYS_PER_YEAR,
)
from models import (
    CurrencyRateDaily,
    EtfHolding,
    HoldingDaily,
    Instrument,
    PortfolioDaily,
    PricesDaily,
)

router = APIRouter()

# Returns window: FX history starts 2024-04-18, so ~2 calendar years is the
# longest GBP-denominated window available anyway.
RETURN_WINDOW_DAYS = 730
# Ledoit-Wolf needs a common (row-complete) window, so one young listing would
# truncate the sample for everyone. ~14 months of daily returns keeps the
# window useful while only dropping genuinely recent IPOs.
MIN_RETURN_OBS = 300
NEW_MONEY_GBP = 1000.0
CACHE_TTL_SECONDS = 3600
# Data-quality guards: forward-filling across a multi-day price hole lumps that
# period's move into one fake "daily" return, inflating variance roughly by the
# hole length. Flag holes longer than a holiday cluster and any single day
# beyond what even meme stocks produce, so the page can warn instead of
# rendering a confidently wrong model.
MAX_FFILL_RUN_DAYS = 5
EXTREME_DAILY_RETURN = 0.60
# Corrupt data (a batch backfill resuming after downtime) makes many symbols
# "move" on the same day; a real melt-up (ETL.PA Mar 2025) is one symbol on
# its own dates. Same-day extremes across this many symbols mean bad inputs.
SUSPECT_SAME_DAY_SYMBOLS = 3
# Positions with avg pairwise correlation above this move as one bet.
GROUP_MIN_CORR = 0.5
# Normal 5% expected shortfall as a multiple of sigma: phi(Phi^-1(0.05)) / 0.05.
# Not fitted to daily tails — a 252-day sum is a CLT object (see the docstring).
ES_5PCT_SIGMA_MULTIPLE = 2.0627

_GBP_FACTORS = {"GBP": 1.0, "GBX": 0.01, "GBp": 0.01}

_cache: dict[str, Any] = {"payload": None, "fetched_at": 0.0}


async def _etf_holdings(session: AsyncSession) -> dict[str, dict[str, float]]:
    """Newest stored snapshot per fund, as {fund symbol: {constituent symbol: pct}}.

    Only constituents that resolved to a tracked instrument come back; the rest
    stay in the fund's residual, and the shortfall is reported as coverage.
    """
    fund, holding = aliased(Instrument), aliased(Instrument)
    latest = (
        select(EtfHolding.etf_instrument_id, func.max(EtfHolding.date).label("date"))
        .group_by(EtfHolding.etf_instrument_id)
        .subquery()
    )
    rows = (
        await session.execute(
            select(fund.yahoo_symbol, holding.yahoo_symbol, EtfHolding.weight_pct)
            .join(latest, (latest.c.etf_instrument_id == EtfHolding.etf_instrument_id)
                  & (latest.c.date == EtfHolding.date))
            .join(fund, fund.id == EtfHolding.etf_instrument_id)
            .join(holding, holding.id == EtfHolding.instrument_id)
        )
    ).all()
    out: dict[str, dict[str, float]] = {}
    for fund_symbol, holding_symbol, pct in rows:
        out.setdefault(fund_symbol, {})[holding_symbol] = (
            out.setdefault(fund_symbol, {}).get(holding_symbol, 0.0) + pct
        )
    return out


async def _load_inputs(session: AsyncSession) -> tuple[date, list, list, list, list, dict]:
    """Fetch the latest holdings snapshot, the monitored universe, price
    history, FX history, and each held ETF's stored holdings."""
    latest_date = (await session.execute(select(func.max(HoldingDaily.date)))).scalar_one()
    holdings = (
        await session.execute(
            select(
                Instrument.t212_code,
                Instrument.name,
                Instrument.currency,
                Instrument.yahoo_symbol,
                Instrument.tags,
                Instrument.thesis,
                HoldingDaily.quantity,
                HoldingDaily.current_price,
            )
            .join(Instrument, Instrument.id == HoldingDaily.instrument_id)
            .where(HoldingDaily.date == latest_date, HoldingDaily.quantity > 0)
        )
    ).all()
    # Full monitored universe (~1k symbols): candidates for the "what would
    # £1k of this do to portfolio vol" figure on the Holdings show-all view.
    instruments = (
        await session.execute(
            select(Instrument.yahoo_symbol, Instrument.currency).where(Instrument.yahoo_symbol.is_not(None))
        )
    ).all()

    start = latest_date - timedelta(days=RETURN_WINDOW_DAYS)
    price_rows = (
        await session.execute(
            select(PricesDaily.symbol, PricesDaily.date, PRICE_COLUMN).where(
                PricesDaily.symbol.in_([i.yahoo_symbol for i in instruments]), PricesDaily.date >= start
            )
        )
    ).all()
    fx_rows = (
        await session.execute(
            select(CurrencyRateDaily.date, CurrencyRateDaily.from_currency, CurrencyRateDaily.rate).where(
                CurrencyRateDaily.to_currency == "GBP", CurrencyRateDaily.date >= start
            )
        )
    ).all()
    return latest_date, holdings, instruments, price_rows, fx_rows, await _etf_holdings(session)


def _fx_frame(fx_rows: list) -> pd.DataFrame:
    """Wide FX frame (index date, one column per source currency), sorted."""
    return (
        pd.DataFrame(fx_rows, columns=["date", "currency", "rate"])
        .pivot(index="date", columns="currency", values="rate")
        .sort_index()
    )


def _gbp_returns(instruments: list, price_rows: list, fx_rows: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide GBP daily-returns frame (index date, one column per yahoo symbol),
    plus a boolean frame marking dates whose price was forward-filled."""
    prices = pd.DataFrame(price_rows, columns=["symbol", "date", "price"]).pivot(
        index="date", columns="symbol", values="price"
    )
    prices = prices.sort_index()
    filled = prices.isna() & prices.ffill().notna()
    prices = prices.ffill()

    fx = _fx_frame(fx_rows).reindex(prices.index).ffill()

    factors = {}
    for inst in instruments:
        if not inst.yahoo_symbol or inst.yahoo_symbol not in prices.columns:
            continue
        if inst.currency in _GBP_FACTORS:
            factors[inst.yahoo_symbol] = pd.Series(_GBP_FACTORS[inst.currency], index=prices.index)
        else:
            factors[inst.yahoo_symbol] = fx.get(inst.currency, pd.Series(np.nan, index=prices.index))
    gbp_prices = prices[list(factors)] * pd.DataFrame(factors)
    return gbp_prices.pct_change(), filled


def _data_warnings(matrix: pd.DataFrame, filled: pd.DataFrame) -> dict[str, Any]:
    """Detect stale-data artifacts inside the model window (see guard constants).

    severity: "high" = inputs look corrupt, the model shouldn't be trusted;
    "info" = extreme but plausibly real single-name moves; None = clean.
    """
    gap_lumped = []
    filled = filled.reindex(matrix.index)
    for s in matrix.columns:
        runs = filled[s]
        max_run = int(runs.groupby((~runs).cumsum()).sum().max())
        if max_run > MAX_FFILL_RUN_DAYS:
            gap_lumped.append({"symbol": s, "max_gap_days": max_run})
    gap_lumped.sort(key=lambda w: w["max_gap_days"], reverse=True)

    stacked = matrix.stack()
    extremes = stacked[stacked.abs() > EXTREME_DAILY_RETURN]
    extreme_returns = [
        {"symbol": sym, "date": d.isoformat(), "value": round(float(v), 3)}
        for (d, sym), v in extremes.items()
    ]
    extreme_returns.sort(key=lambda w: abs(w["value"]), reverse=True)

    per_date: dict[str, int] = {}
    for w in extreme_returns:
        per_date[w["date"]] = per_date.get(w["date"], 0) + 1
    if gap_lumped or max(per_date.values(), default=0) >= SUSPECT_SAME_DAY_SYMBOLS:
        severity = "high"
    elif extreme_returns:
        severity = "info"
    else:
        severity = None
    return {"gap_lumped": gap_lumped[:20], "extreme_returns": extreme_returns[:20], "severity": severity}


def _breakeven_excess_return(beta: float, ann_vol: float) -> float:
    """Extra return an asset must earn over the portfolio to be worth adding.

    A position improves compounded (geometric) growth when the return it adds
    beats the variance drag it adds: d_mu > d(sigma^2)/2. In the limit of a small
    addition that reduces to sigma^2 * (beta - 1), so the hurdle is a property of
    the asset and does not move with the size of the cheque.

    Negative means the asset may underperform the portfolio and still help.
    """
    return float(ann_vol**2 * (beta - 1.0))


def _candidate_vol_deltas(
    returns: pd.DataFrame,
    matrix: pd.DataFrame,
    weights: np.ndarray,
    port_var: float,
    ann_vol: float,
    model_value: float,
    held: set[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Δvol (bp) and breakeven excess return for each non-held monitored symbol.

    Closed form: the new position only adds a cov(candidate, portfolio) cross
    term and its own variance, so no covariance-matrix rebuild is needed.
    Cross-moments are sample estimates (the candidate is outside the shrunk
    matrix); the base variance is the shrunk one so figures line up with the
    held positions'.
    """
    port_returns = pd.Series(matrix.values @ weights, index=matrix.index)
    v, a = model_value, NEW_MONEY_GBP
    out: dict[str, float] = {}
    breakeven: dict[str, float] = {}
    for sym in returns.columns:
        if sym in held:
            continue
        series = returns[sym].reindex(matrix.index)
        mask = series.notna()
        if int(mask.sum()) < MIN_RETURN_OBS:
            continue
        c = series[mask].values
        p = port_returns[mask].values
        cov_cp = float(np.mean((c - c.mean()) * (p - p.mean())))
        var_c = float(c.var())
        new_var = (v**2 * port_var + 2 * v * a * cov_cp + a**2 * var_c) / (v + a) ** 2
        out[sym] = round(float(np.sqrt(new_var * TRADING_DAYS_PER_YEAR) - ann_vol) * 1e4, 2)
        breakeven[sym] = round(_breakeven_excess_return(cov_cp / port_var, ann_vol), 4)
    return out, breakeven


def _thesis_risk_band(raw: Any, beta: float) -> dict[str, Any]:
    """The investor's conviction and target weight band, restated in risk terms.

    The bands in Instrument.thesis are authored as shares of *capital*, but a
    position's share of *risk* is beta times its weight — so a name can sit
    inside its band on money and far outside it on risk. Converting with the
    same beta the hurdle uses makes the two comparable.

    thesis_rules.py remains the authority that enforces the capital band; this
    is a diagnostic view of it.
    """
    blank = {"conviction": None, "band_risk_min": None, "band_risk_max": None}
    if not raw:
        return blank
    try:
        thesis = InstrumentThesisSchema.model_validate(raw)
    except ValidationError as e:
        logger.info("Invalid thesis, omitted from risk bands: %s", e.errors())
        return blank
    return {
        "conviction": thesis.conviction,
        "band_risk_min": thesis.target_weight_min_pct / 100.0 * beta,
        "band_risk_max": thesis.target_weight_max_pct / 100.0 * beta,
    }


def _look_through(
    returns: pd.DataFrame,
    etf_holdings: dict[str, dict[str, float]],
    included: list[str],
    values: np.ndarray,
    dates: pd.Index,
) -> dict[str, Any] | None:
    """Risk attribution with each fund expanded into its published constituents.

    Answers a different question from the line-item view: not "which ticker do I
    trade" but "which company am I actually exposed to". A fund's enumerated
    weight moves to the underlying names; its un-enumerated tail stays on the
    fund's own return series.

    Deliberately reports no effective-bets figure: enumerating an index tail adds
    hundreds of sub-0.01% exposures that inflate an entropy measure without
    representing decisions, so it would not be comparable to the line-item one.

    Runs on the line-item model's own dates: a constituent that does not cover
    them is left in the residual rather than allowed to truncate the panel. One
    recent listing among hundreds would otherwise drop the shared history from
    501 observations to 301 and quietly weaken every correlation in the matrix.

    That residual proxy carries whole-fund beta rather than tail beta, so it
    slightly overstates correlation and prints a higher portfolio vol than the
    line-item model. The reattribution is sound; the vol gap is an artifact and
    must be labelled as one, never shown as the truer number.

    Returns None when no held fund has constituent data.
    """
    exposures: dict[str, float] = {}
    funds = []
    for sym, value in zip(included, values):
        spec = etf_holdings.get(sym)
        if spec is None:
            exposures[sym] = exposures.get(sym, 0.0) + float(value)
            continue
        enumerated = 0.0
        for name, pct in spec.items():
            if name not in returns.columns or returns[name].reindex(dates).isna().any():
                continue  # cannot cover the panel: leave this slice in the residual
            slice_gbp = float(value) * pct / 100.0
            if slice_gbp < LOOKTHROUGH_MIN_EXPOSURE_GBP:
                continue  # immaterial, and each one costs a degree of freedom
            exposures[name] = exposures.get(name, 0.0) + slice_gbp
            enumerated += pct
        exposures[sym] = exposures.get(sym, 0.0) + float(value) * (100.0 - enumerated) / 100.0
        funds.append({"symbol": sym, "enumerated_pct": round(enumerated, 2)})
    if not funds:
        return None

    symbols = sorted(exposures)
    matrix = returns[symbols].reindex(dates)
    if matrix.isna().to_numpy().any() or matrix.shape[0] < MIN_RETURN_OBS:
        return None

    cov, _ = ledoit_wolf_cov(matrix.values)
    w = np.array([exposures[s] for s in symbols])
    w = w / w.sum()
    rc, port_var = risk_contributions(cov, w)
    ann_vol = float(np.sqrt(port_var * TRADING_DAYS_PER_YEAR))

    held = {s for s in included if s not in etf_holdings}
    rows = sorted(
        (
            {
                "symbol": s,
                "weight": float(w[i]),
                "risk_contribution": float(rc[i]),
                "held_directly": s in held,
            }
            for i, s in enumerate(symbols)
        ),
        key=lambda r: r["risk_contribution"],
        reverse=True,
    )
    return {
        "funds": funds,
        "n_obs": int(matrix.shape[0]),
        "n_assets": len(symbols),
        "annual_volatility": ann_vol,
        "residual_risk_share": float(sum(rc[symbols.index(f["symbol"])] for f in funds)),
        "exposures": rows,
    }


def build_risk_payload(
    latest_date: date,
    holdings: list,
    instruments: list,
    price_rows: list,
    fx_rows: list,
    etf_holdings: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Assemble the /api/risk payload. Pure (no I/O) for standalone testing."""
    returns, filled = _gbp_returns(instruments, price_rows, fx_rows)

    # Aggregate holding values per yahoo symbol (t212 codes map 1:1 today, but
    # the schema doesn't guarantee it).
    info: dict[str, Any] = {}
    for h in holdings:
        sym = h.yahoo_symbol
        if sym in info:
            info[sym]["quantity"] += h.quantity
        else:
            info[sym] = {
                "t212_code": h.t212_code,
                "name": h.name,
                "currency": h.currency,
                "tags": h.tags or [],
                "thesis": h.thesis,
                "quantity": h.quantity,
                "current_price": h.current_price,
            }

    # Latest GBP value per holding: T212 live price × latest FX rate.
    fx_last = _fx_frame(fx_rows).ffill().iloc[-1]
    for sym, d in info.items():
        rate = _GBP_FACTORS.get(d["currency"], fx_last.get(d["currency"]))
        d["value_gbp"] = float(d["quantity"] * d["current_price"] * (rate if rate is not None else np.nan))

    total_value = float(sum(d["value_gbp"] for d in info.values() if pd.notna(d["value_gbp"])))

    included = [s for s in returns.columns if returns[s].count() >= MIN_RETURN_OBS and s in info]
    excluded = [
        {
            "symbol": s,
            "t212_code": d["t212_code"],
            "name": d["name"],
            "value_gbp": None if pd.isna(d["value_gbp"]) else round(d["value_gbp"], 2),
            "reason": (
                f"only {int(returns[s].count()) if s in returns.columns else 0} daily returns "
                f"(need {MIN_RETURN_OBS})"
            ),
        }
        for s, d in info.items()
        if s not in included
    ]

    matrix = returns[included].dropna()
    cov, shrinkage = ledoit_wolf_cov(matrix.values)

    values = np.array([info[s]["value_gbp"] for s in included])
    model_value = float(values.sum())
    weights = values / model_value

    rc, port_var = risk_contributions(cov, weights)
    ann_vol = float(np.sqrt(port_var * TRADING_DAYS_PER_YEAR))
    hrp = hrp_weights(cov)

    # Benchmark from the full universe frame — it need not be a holding.
    benchmark_vol = None
    if SPY in returns.columns:
        bench = returns[SPY].reindex(matrix.index).dropna()
        if len(bench) >= MIN_RETURN_OBS:
            benchmark_vol = float(bench.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

    # Correlated groups: positions that statistically move as one bet.
    sample_corr = np.corrcoef(matrix.values.T)
    groups = []
    single_count = 0
    for idx in correlation_clusters(cov, GROUP_MIN_CORR):
        if len(idx) < 2:
            single_count += 1
            continue
        pair_corrs = [sample_corr[i, j] for i in idx for j in idx if i < j]
        groups.append(
            {
                "symbols": [included[i] for i in idx],
                "risk_share": float(rc[idx].sum()),
                "value_weight": float(weights[idx].sum()),
                "avg_corr": float(np.mean(pair_corrs)),
            }
        )
    groups.sort(key=lambda g: g["risk_share"], reverse=True)

    # Volatility impact of putting NEW_MONEY_GBP into each position.
    vol_deltas = []
    for i in range(len(included)):
        new_values = values.copy()
        new_values[i] += NEW_MONEY_GBP
        w_new = new_values / new_values.sum()
        _, var_new = risk_contributions(cov, w_new)
        vol_deltas.append((np.sqrt(var_new * TRADING_DAYS_PER_YEAR) - ann_vol) * 1e4)

    holdings_payload = sorted(
        (
            {
                "symbol": s,
                "t212_code": info[s]["t212_code"],
                "name": info[s]["name"],
                "value_gbp": round(info[s]["value_gbp"], 2),
                "weight": float(weights[i]),
                "risk_contribution": float(rc[i]),
                "beta_to_portfolio": float(rc[i] / weights[i]),
                "breakeven_excess_return": round(
                    _breakeven_excess_return(rc[i] / weights[i], ann_vol), 4
                ),
                "hrp_weight": float(hrp[i]),
                **_thesis_risk_band(info[s]["thesis"], rc[i] / weights[i]),
                "next_1k_vol_delta_bp": round(float(vol_deltas[i]), 2),
                "tags": info[s]["tags"],
            }
            for i, s in enumerate(included)
        ),
        key=lambda h: h["risk_contribution"],
        reverse=True,
    )

    candidate_vol_deltas, candidate_breakeven = _candidate_vol_deltas(
        returns, matrix, weights, port_var, ann_vol, model_value, set(info)
    )

    rc_by_symbol = dict(zip(included, rc))
    clusters = []
    for tag, cap in TAG_CLUSTER_SOFT_CAPS.items():
        tagged = [s for s in info if tag in info[s]["tags"]]
        if not tagged:
            continue
        value_weight = float(
            sum(info[s]["value_gbp"] for s in tagged if pd.notna(info[s]["value_gbp"])) / total_value
        )
        # What the buy gate actually tests: ETFs excluded from thematic clusters.
        direct = [s for s in tagged if tag == ETF_TAG or ETF_TAG not in info[s]["tags"]]
        gated_weight = float(
            sum(info[s]["value_gbp"] for s in direct if pd.notna(info[s]["value_gbp"])) / total_value
        )
        risk_share = float(sum(rc_by_symbol[s] for s in tagged if s in rc_by_symbol))
        alert_pct = TAG_RISK_ALERT_PCT.get(tag)
        clusters.append(
            {
                "tag": tag,
                "value_weight": value_weight,
                "gated_weight": gated_weight,
                "risk_share": risk_share,
                "soft_cap": cap / 100.0,
                "risk_alert": None if alert_pct is None else alert_pct / 100.0,
                "risk_alert_breached": alert_pct is not None and risk_share >= alert_pct / 100.0,
            }
        )
    clusters.sort(key=lambda c: c["risk_share"], reverse=True)

    # Which holdings are closed to new money. Must mirror _cluster_headroom_gbp
    # exactly, or the page promises what the agent will veto.
    breached = {c["tag"] for c in clusters if c["gated_weight"] >= c["soft_cap"]}
    for h in holdings_payload:
        is_etf = ETF_TAG in h["tags"]
        h["blocked_tags"] = sorted(
            t for t in breached.intersection(h["tags"]) if t == ETF_TAG or not is_etf
        )

    return {
        "as_of": latest_date.isoformat(),
        "window": {
            "start": matrix.index[0].isoformat(),
            "end": matrix.index[-1].isoformat(),
            "n_obs": int(matrix.shape[0]),
            "n_assets": len(included),
        },
        "portfolio": {
            "total_value_gbp": round(total_value, 2),
            "model_value_gbp": round(model_value, 2),
            "annual_volatility": ann_vol,
            "benchmark_symbol": SPY,
            "benchmark_volatility": benchmark_vol,
            "effective_bets": effective_bets(cov, weights),
            # Correlation-blind: how evenly risk spreads across line items,
            # which is a different question from how many bets there are.
            "risk_concentration_bets": float(1.0 / (rc**2).sum()),
            # What volatility costs a compounder: g = mu - sigma^2/2.
            "variance_drag": ann_vol**2 / 2.0,
            "expected_shortfall_5pct": ES_5PCT_SIGMA_MULTIPLE * ann_vol,
            "max_tolerable_annual_loss": MAX_TOLERABLE_ANNUAL_LOSS,
            "shrinkage": float(shrinkage),
            "new_money_gbp": NEW_MONEY_GBP,
        },
        # Tags closed to new money. The single source of truth for every surface
        # that shows cap state, so no page can drift from the agent's gate.
        "capped_tags": sorted(breached),
        "groups": {"correlated": groups, "independent_count": single_count, "min_corr": GROUP_MIN_CORR},
        "candidates": candidate_vol_deltas,
        "candidate_breakeven": candidate_breakeven,
        "holdings": holdings_payload,
        "clusters": clusters,
        "look_through": _look_through(returns, etf_holdings, included, values, matrix.index),
        "excluded": excluded,
        "warnings": _data_warnings(matrix, filled),
    }


async def _risk_adjusted_stats(session: AsyncSession) -> dict[str, Any]:
    """Latest stored risk-adjusted returns, to sit beside the modelled volatility.

    Volatility alone says how much risk is being run, not whether it is being
    paid for; these are the fields that answer that. Read from the newest row
    that has them rather than the newest row outright, since they are written by
    a separate job (scripts/update_returns.py) and can lag by a day.
    """
    row = (
        await session.execute(
            select(
                PortfolioDaily.date,
                PortfolioDaily.sharpe_ratio,
                PortfolioDaily.sortino_ratio,
                PortfolioDaily.jensens_alpha,
                PortfolioDaily.beta,
                PortfolioDaily.twrr,
            )
            .where(PortfolioDaily.sharpe_ratio.is_not(None))
            .order_by(PortfolioDaily.date.desc())
            .limit(1)
        )
    ).first()
    if row is None:
        return {}
    return {
        "as_of": row.date.isoformat(),
        "sharpe_ratio": row.sharpe_ratio,
        "sortino_ratio": row.sortino_ratio,
        "jensens_alpha": row.jensens_alpha,
        "beta": row.beta,
        # Stored as a percent (18.04 = 18.04%, scripts/update_returns.py); the
        # shortfall card does arithmetic with it, so normalise at the boundary.
        "twrr": None if row.twrr is None else row.twrr / 100.0,
    }


@router.get("/api/risk")
async def get_risk(session: AsyncSession = Depends(get_db_session)) -> dict[str, Any]:
    now = time.time()
    if _cache["payload"] is not None and now - _cache["fetched_at"] < CACHE_TTL_SECONDS:
        return _cache["payload"]
    payload = build_risk_payload(*await _load_inputs(session))
    payload["risk_adjusted"] = await _risk_adjusted_stats(session)
    _cache.update(payload=payload, fetched_at=now)
    return payload
