"""
FastAPI backend for Trading212 Portfolio Manager
"""

# Standard library imports
import asyncio
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, TypedDict

# Third-party imports
import aiohttp
import numpy as np
from dateutil.relativedelta import relativedelta
from fastapi import Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

# Local imports
from backend.app import app, get_db_session
from backend.screener_config import get_screener_config
from backend.utils.dcf import get_dcf_prices
from backend.utils.market_data import (
    gen_buffett_indicator,
    gen_fear_greed_index,
    gen_market_breadth_indicator,
    gen_sp500_above_sma200,
    get_consumer_sentiment,
    get_yield_spread,
)
from backend.utils.roic import get_roic
from backend.utils.screener import calculate_screener_results
from backend.utils.technical import calculate_technical_indicators_for_symbols
from config import BENCHES, CURRENCIES, PRICE_FIELD, TIMEZONE, VIX
from data import QUICK_RATIO_THRESHOLDS
from models import (
    CurrencyRateDaily,
    Form13FFiling,
    Form13FHolding,
    Form13FManager,
    HoldingDaily,
    Instrument,
    InstrumentMetricsDaily,
    Pie,
    PieInstrument,
    PortfolioDaily,
    PricesDaily,
    TransactionHistory,
)

PRICE_COLUMN = getattr(PricesDaily, PRICE_FIELD.lower().replace(" ", "_") + "_price").label("price")

# 13F score thresholds (USD for value, % for change)
FORM13F_MIN_VALUE_NEW = 50_000  # Ignore "New" positions below this (pilot/noise)
FORM13F_INCREASE_EFFECTIVE_NEW = 1000  # +1000%+ = effectively new position
FORM13F_TRIM_EFFECTIVE_LIQUIDATION = -90  # -90%+ = effectively liquidated


class Form13FHolder(TypedDict):
    """13F holder with change and optional report data."""

    manager_id: int | None
    name: str
    change: str
    report_date: str | None
    shares: int | None
    shares_prev: int | None
    value: int | None
    scored: bool
    score_reason: str | None


class Form13FInstrumentResult(TypedDict):
    """13F score and holders for one instrument."""

    score: float
    holders: list[Form13FHolder]


class Form13FFilingRow(TypedDict):
    """Aggregated row per (manager, filing) for a single instrument in get_instrument."""

    manager_name: str
    manager_id: int
    manager_cik: str
    report_date: date
    accession_number: str
    value: int
    shares: int
    filing_total_value: int


async def get_rates(session: AsyncSession) -> dict[str, float]:
    """Get current currency exchange rates to GBP."""
    table = {"GBX": 0.01, "GBP": 1.0, "GBp": 0.01}

    result = await session.execute(
        select(CurrencyRateDaily.from_currency, CurrencyRateDaily.rate).filter(
            CurrencyRateDaily.from_currency.in_(CURRENCIES),
            CurrencyRateDaily.to_currency == "GBP",
            CurrencyRateDaily.date == datetime.now(TIMEZONE).date(),
        )
    )
    rates = result.all()
    for currency, rate in rates:
        table[currency] = rate

    return table


def calculate_historical_trends(holding: HoldingDaily) -> dict[str, Optional[float]]:
    """Calculates trend metrics from historical data stored in the yahoo object"""
    trends: dict[str, Optional[float]] = {
        "recommendation_trend": None,
        "pe_1y_trend_pct": None,
        "pe_5y_avg_vs_current_pct": None,
    }

    if holding.instrument.yahoo is None:
        return trends

    # --- 1. Recommendation Trend ---
    recs = holding.instrument.yahoo.recommendations
    trends["recommendation_trend"] = 0.0
    if recs and len(recs) >= 2:
        items = sorted(recs.items())
        sb = np.array([m.get("strongBuy", 0) for _, m in items], dtype=float)
        b = np.array([m.get("buy", 0) for _, m in items], dtype=float)
        h = np.array([m.get("hold", 0) for _, m in items], dtype=float)
        s = np.array([m.get("sell", 0) for _, m in items], dtype=float)
        ss = np.array([m.get("strongSell", 0) for _, m in items], dtype=float)
        tot = sb + b + h + s + ss
        mask = tot > 0
        if mask.sum() >= 2:
            score = (2 * sb + b - s - 2 * ss) / (2 * tot)
            score = score[mask]
            x = np.arange(score.size, dtype=float)
            if score.std() != 0:
                trends["recommendation_trend"] = float(np.corrcoef(x, score)[0, 1])

    # --- 2. PE Trend and PE vs History ---
    pes = holding.instrument.yahoo.pes
    current_pe = holding.instrument.yahoo.info.get("trailingPE")

    if pes and current_pe and current_pe > 0:
        one_year_ago = (datetime.now() - relativedelta(years=1)).strftime("%Y-%m-%d")
        five_years_ago = (datetime.now() - relativedelta(years=5)).strftime("%Y-%m-%d")

        # PE 1 Year Trend
        past_pe_date = next((d for d in sorted(pes.keys(), reverse=True) if d <= one_year_ago), None)
        if past_pe_date and pes[past_pe_date].get("pe_ratio", 0) > 0:
            past_pe = pes[past_pe_date]["pe_ratio"]
            trends["pe_1y_trend_pct"] = (current_pe / past_pe - 1) * 100

        # PE vs 5Y Average
        pe_values_5y = [v["pe_ratio"] for k, v in pes.items() if k >= five_years_ago and v.get("pe_ratio", 0) > 0]
        if pe_values_5y:
            avg_pe_5y = sum(pe_values_5y) / len(pe_values_5y)
            trends["pe_5y_avg_vs_current_pct"] = (avg_pe_5y / current_pe - 1) * 100

    return trends


@app.get("/api/portfolio/current")
async def get_current_portfolio(session: AsyncSession = Depends(get_db_session), show_all: bool = False) -> dict[str, Any]:
    """Get current portfolio holdings with detailed information. show_all=true: all monitored instruments."""
    # Get the latest snapshot date
    result = await session.execute(select(func.max(HoldingDaily.date)))
    latest_date = result.scalar()

    if show_all:
        # All monitored instruments: real holdings where held, mock (quantity=0) for others
        instruments_result = await session.execute(
            select(Instrument)
            .where(Instrument.yahoo_symbol.isnot(None))
            .order_by(Instrument.name)
            .options(selectinload(Instrument.yahoo))
        )
        instruments = list(instruments_result.scalars().all())
        if not instruments:
            return {"holdings": [], "total_holdings": 0, "quick_ratio_thresholds": QUICK_RATIO_THRESHOLDS, "last_updated": None}
        holdings_result = await session.execute(
            select(HoldingDaily)
            .join(Instrument)
            .filter(HoldingDaily.date == latest_date)
            .options(selectinload(HoldingDaily.instrument).selectinload(Instrument.yahoo))
        )
        holdings_list = holdings_result.scalars().all()
        holding_by_symbol = {(h.instrument.yahoo_symbol or h.instrument.t212_code): h for h in holdings_list}
        symbols_held = set(holding_by_symbol.keys())
        currency_rates = await get_rates(session)
        total_portfolio_value = sum(
            h.quantity * h.current_price * currency_rates.get(h.instrument.currency, 1.0)
            for h in holdings_list
        )
        items = []
        for inst in instruments:
            sym = inst.yahoo_symbol
            if not sym:
                continue
            if sym in symbols_held:
                items.append(holding_by_symbol[sym])
            else:
                info = (inst.yahoo.info or {}) if inst.yahoo else {}
                price = (info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")) or 0
                if not isinstance(price, (int, float)):
                    price = 0
                items.append(SimpleNamespace(
                    instrument=inst,
                    quantity=0,
                    avg_price=None,
                    current_price=float(price) if price else 0,
                    ppl=None,
                    fx_ppl=None,
                    date=latest_date or datetime.now(TIMEZONE).date(),
                ))
        symbols_for_technical = [i.yahoo_symbol for i in instruments if i.yahoo_symbol]
        instruments_for_dcf = instruments
    else:
        if not latest_date:
            return {"holdings": [], "total_holdings": 0, "quick_ratio_thresholds": QUICK_RATIO_THRESHOLDS, "last_updated": None}
        # Query holdings with instrument data in the same session
        holdings_result = await session.execute(
            select(HoldingDaily)
            .join(Instrument)
            .filter(HoldingDaily.date == latest_date)
            .order_by(Instrument.name)
            .options(selectinload(HoldingDaily.instrument).selectinload(Instrument.yahoo))
        )
        holdings = holdings_result.scalars().all()
        # Get currency rates
        currency_rates = await get_rates(session)
        total_portfolio_value = 0.0
        for holding in holdings:
            market_value_native = holding.quantity * holding.current_price
            market_value_gbp = market_value_native * currency_rates[holding.instrument.currency]
            total_portfolio_value += market_value_gbp
        items = holdings
        symbols_for_technical = [h.instrument.yahoo_symbol for h in holdings if h.instrument.yahoo_symbol]
        instruments_for_dcf = [h.instrument for h in holdings]

    # Calculate technical indicators using centralized function
    rsi_data, technical_data = await calculate_technical_indicators_for_symbols(symbols_for_technical, session)
    dcf_prices = await get_dcf_prices(instruments_for_dcf)
    dcf_prices_dict = dict(zip(symbols_for_technical, dcf_prices))

    instrument_ids = [h.instrument.id for h in items]
    form13f = await _get_form13f_for_instruments(session, instrument_ids)

    portfolio_data = []
    for holding in items:
        market_value_native = holding.quantity * holding.current_price
        market_value_gbp = market_value_native * currency_rates.get(holding.instrument.currency, 1.0)
        portfolio_pct = (market_value_gbp / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
        dcf_price = dcf_prices_dict.get(holding.instrument.yahoo_symbol)

        # Yahoo Finance info for this instrument
        info = (holding.instrument.yahoo.info or {}) if holding.instrument.yahoo else {}
        trends = calculate_historical_trends(holding)
        profit = holding.ppl if holding.ppl is not None else 0
        cost_basis = (market_value_gbp - holding.ppl) if holding.ppl is not None else 0
        return_pct = (holding.ppl / cost_basis * 100.0) if holding.ppl is not None and cost_basis > 0 else 0.0

        portfolio_data.append(
            {
                "t212_code": holding.instrument.t212_code,
                "name": holding.instrument.name,
                "yahoo_symbol": holding.instrument.yahoo_symbol,
                "currency": holding.instrument.currency,
                "sector": info.get("sector"),
                "country": info.get("country"),
                "quantity": holding.quantity,
                "avg_price": holding.avg_price,
                "current_price": holding.current_price,
                "analyst_price_targets": holding.instrument.yahoo.analyst_price_targets if holding.instrument.yahoo else None,
                "dcf_price": dcf_price,
                "dcf_diff": (dcf_price / holding.current_price - 1) if (dcf_price and holding.current_price) else None,
                "ppl": holding.ppl,
                "fx_ppl": holding.fx_ppl,
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "ps_ratio": info.get("priceToSalesTrailing12Months"),
                "avg_pe": getattr(holding.instrument.yahoo, "avg_pe_5y", None) if holding.instrument.yahoo else None,
                "beta": info.get("beta"),
                "date": holding.date.isoformat() if hasattr(holding.date, "isoformat") else str(holding.date),
                "market_value": market_value_gbp,  # Now in GBP
                "profit": profit,  # Total profit (same as terminal - ppl already includes FX)
                "return_pct": return_pct,
                "portfolio_pct": portfolio_pct,
                "dividend_yield": info.get("dividendYield"),
                "business_summary": info.get("longBusinessSummary"),
                "prediction": (info["targetMedianPrice"] / holding.current_price - 1) * 100.0
                if info.get("targetMedianPrice")
                else None,
                "institutional_ownership": round(info.get("heldPercentInstitutions") * 100.0)
                if (info.get("heldPercentInstitutions") is not None)
                else None,
                "peg_ratio": info["trailingPegRatio"]
                if info.get("trailingPegRatio")
                else None,  # Keep full precision for screener evaluation
                "profit_margins": info["profitMargins"] * 100.0
                if info.get("profitMargins")
                else None,  # Keep full precision for screener evaluation
                "revenue_growth": info["revenueGrowth"] * 100.0
                if info.get("revenueGrowth")
                else None,  # Keep full precision for screener evaluation
                "return_on_assets": info["returnOnAssets"] * 100.0
                if info.get("returnOnAssets")
                else None,  # Keep full precision for screener evaluation
                "return_on_equity": info["returnOnEquity"] * 100.0
                if info.get("returnOnEquity")
                else None,  # Keep full precision for screener evaluation
                "roic": get_roic(info),
                "free_cashflow_yield": info["freeCashflow"] / info["marketCap"] * 100
                if (info.get("freeCashflow") and info.get("marketCap", 0) > 0)
                else None,
                "quickRatio": info.get("quickRatio")
                if info.get("sector") != "Financial Services"
                else None,
                "debtToEquity": info.get("debtToEquity"),
                "recommendation_mean": round(info["recommendationMean"], 2) if info.get("recommendationMean") else None,
                "recommendation_key": info.get("recommendationKey"),
                "recommendations": holding.instrument.yahoo.recommendations if holding.instrument.yahoo else None,
                "number_of_analyst_opinions": info.get("numberOfAnalystOpinions"),
                "fifty_two_week_high_distance": round(info["fiftyTwoWeekHighChangePercent"] * 100)
                if info.get("fiftyTwoWeekHighChangePercent")
                else None,  # Distance from 52-week high (negative = below high)
                "fifty_two_week_change": round(info.get("52WeekChange", 0) * 100)
                if info.get("52WeekChange") is not None
                else None,
                "short_percent_of_float": info["shortPercentOfFloat"] * 100
                if info.get("shortPercentOfFloat")
                else None,
                "rsi": rsi_data.get(holding.instrument.yahoo_symbol),
                "rule_of_40_score": (info.get("revenueGrowth", 0) * 100) + (info.get("profitMargins", 0) * 100)
                if (info.get("revenueGrowth") is not None and info.get("profitMargins") is not None)
                else None,  # Keep full precision
                # Technical indicators calculated from price history
                "sma_20": technical_data.get(holding.instrument.yahoo_symbol, {}).get("sma_20"),
                "sma_50": technical_data.get(holding.instrument.yahoo_symbol, {}).get("sma_50"),
                "sma_200": technical_data.get(holding.instrument.yahoo_symbol, {}).get("sma_200"),
                "rs_6m_vs_spy": technical_data.get(holding.instrument.yahoo_symbol, {}).get("rs_6m_vs_spy"),
                "gc_days_since": technical_data.get(holding.instrument.yahoo_symbol, {}).get("gc_days_since"),
                "gc_within_sma50_frac": technical_data.get(holding.instrument.yahoo_symbol, {}).get(
                    "gc_within_sma50_frac"
                ),
                "bb_width_20": technical_data.get(holding.instrument.yahoo_symbol, {}).get("bb_width_20"),
                "bb_width_20_p30_6m": technical_data.get(holding.instrument.yahoo_symbol, {}).get("bb_width_20_p30_6m"),
                "vol20_lt_vol60": technical_data.get(holding.instrument.yahoo_symbol, {}).get("vol20_lt_vol60"),
                "volume_ratio": technical_data.get(holding.instrument.yahoo_symbol, {}).get("volume_ratio"),
                **trends,
                "quote_type": info.get("quoteType", "Unknown"),
                "passedScreeners": [],  # will be populated below
                "screener_score": 0,  # will be populated below
                "form13f_score": form13f.get(holding.instrument.id, {}).get("score"),
                "form13f_holders": form13f.get(holding.instrument.id, {}).get("holders", []),
            }
        )

    # Calculate screener results for each holding
    calculate_screener_results(portfolio_data)

    last_updated = latest_date.isoformat() if latest_date else datetime.now(TIMEZONE).date().isoformat()
    return {
        "holdings": portfolio_data,
        "total_holdings": len(portfolio_data),
        "quick_ratio_thresholds": QUICK_RATIO_THRESHOLDS,
        "last_updated": last_updated,
    }


@app.get("/api/portfolio/summary")
async def get_portfolio_summary(session: AsyncSession = Depends(get_db_session)) -> dict[str, Any]:
    """Get portfolio summary statistics including total value, profit, and win rate."""
    # Get the latest portfolio snapshot
    result = await session.execute(select(PortfolioDaily).order_by(PortfolioDaily.date.desc()))
    latest_snapshot = result.scalars().first()

    if not latest_snapshot:
        return {"error": "No portfolio data available"}

    # Get holdings for the same date to calculate win rate
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=False), raise_for_status=True
    ) as aiohttp_session:
        (
            holdings_result,
            vix_result,
            fear_greed_index,
            yield_spread,
            buffett_indicator,
            market_breadth_indicator,
            sp500_above_sma200,
            consumer_sentiment,
        ) = await asyncio.gather(
            session.execute(select(HoldingDaily).filter(HoldingDaily.date == latest_snapshot.date)),
            session.execute(
                select(PricesDaily.close_price)
                .where(
                    PricesDaily.symbol == VIX,
                )
                .order_by(PricesDaily.date.desc())
                .limit(1)
            ),
            gen_fear_greed_index(aiohttp_session),
            get_yield_spread(aiohttp_session),
            gen_buffett_indicator(aiohttp_session),
            gen_market_breadth_indicator(session, aiohttp_session),
            gen_sp500_above_sma200(session),
            get_consumer_sentiment(aiohttp_session),
        )

    holdings = holdings_result.scalars().all()

    profitable_count = 0
    losing_count = 0

    for holding in holdings:
        profit = holding.ppl  # Total profit (same as terminal - ppl already includes FX)
        if profit > 0:
            profitable_count += 1
        else:
            losing_count += 1

    return {
        "total_value": latest_snapshot.value,
        "total_profit": latest_snapshot.unrealised_profit,
        "total_return_pct": latest_snapshot.return_pct,
        "total_holdings": len(holdings),
        "profitable_holdings": profitable_count,
        "losing_holdings": losing_count,
        "beta": latest_snapshot.beta,
        "sortino_ratio": latest_snapshot.sortino_ratio,
        "sharpe_ratio": latest_snapshot.sharpe_ratio,
        "mwrr": latest_snapshot.mwrr,
        "twrr": latest_snapshot.twrr,
        "market_breadth_indicator": market_breadth_indicator,
        "sp500_above_sma200": sp500_above_sma200,
        "last_updated": latest_snapshot.updated_at.isoformat(),
        "vix": vix_result.scalar(),
        "fear_greed_index": fear_greed_index,
        "yield_spread": yield_spread,
        "buffett_indicator": buffett_indicator,
        "consumer_sentiment": consumer_sentiment,
    }


@app.get("/api/portfolio/allocations")
async def get_portfolio_allocations(session: AsyncSession = Depends(get_db_session)) -> dict[str, Any]:
    """Get portfolio allocations by sector and country."""
    # Get the latest portfolio snapshot
    result = await session.execute(select(PortfolioDaily).order_by(PortfolioDaily.date.desc()))
    latest_snapshot = result.scalars().first()

    if not latest_snapshot:
        return {"error": "No portfolio data available"}

    return {
        "sector_allocation": latest_snapshot.sector_allocation,
        "country_allocation": latest_snapshot.country_allocation,
        "currency_allocation": latest_snapshot.currency_allocation,
        "etf_equity_split": latest_snapshot.etf_equity_split,
        "total_value": latest_snapshot.value,
    }


@app.get("/api/portfolio/history")
async def get_portfolio_history(
    days: Optional[int] = None, session: AsyncSession = Depends(get_db_session)
) -> dict[str, Any]:
    """Get portfolio history for the last N days."""
    if days is not None:
        cutoff_date = datetime.now(TIMEZONE).date() - timedelta(days=days)
        query = select(PortfolioDaily).filter(PortfolioDaily.date >= cutoff_date).order_by(PortfolioDaily.date)
    else:
        query = select(PortfolioDaily).order_by(PortfolioDaily.date)

    snap_res = await session.execute(query)
    snapshots = snap_res.scalars().all()

    bench_res = await session.execute(
        select(PricesDaily.date, PRICE_COLUMN, PricesDaily.symbol)
        .where(
            PricesDaily.symbol.in_(BENCHES),
            PricesDaily.date >= snapshots[0].date,  # latest date in the snapshots
        )
        .order_by(PricesDaily.date)
    )
    bench_rows = bench_res.all()
    daily_bench: dict[str, dict[date, float]] = defaultdict(lambda: defaultdict(float))
    bench_prices = defaultdict(list)
    for bench_row in bench_rows:
        daily_bench[bench_row.symbol][bench_row.date] = bench_row.price
        bench_prices[bench_row.symbol].append(bench_row.price)
    benches_base_price = {symbol: bench_price[0] for symbol, bench_price in bench_prices.items()}
    bench_start = snapshots[0].return_pct
    bench = [bench_start for _bench_symbol in BENCHES]

    history_data = []
    for snapshot in snapshots:
        bench = [
            bench_start + (daily_bench[bench_symbol][snapshot.date] / benches_base_price[bench_symbol] - 1) * 100
            if daily_bench[bench_symbol].get(snapshot.date)
            else bench[i]
            for i, bench_symbol in enumerate(BENCHES)
        ]
        history_data.append(
            {
                "date": snapshot.date.isoformat(),
                "total_value": snapshot.value,
                "total_profit": snapshot.unrealised_profit,
                "total_return_pct": snapshot.return_pct,
                "country_allocation": snapshot.country_allocation,
                "sector_allocation": snapshot.sector_allocation,
                "currency_allocation": snapshot.currency_allocation,
                "etf_equity_split": snapshot.etf_equity_split,
                "benchmark_return_pct": bench,
            }
        )

    return {"history": history_data, "days": days, "benchmark": BENCHES}


@app.get("/api/tickers")
async def get_instruments(session: AsyncSession = Depends(get_db_session)) -> dict[str, list[dict[str, Any]]]:
    """Get all instruments in the database for autocomplete."""
    name_col = func.coalesce(Instrument.name, PricesDaily.symbol)
    result = await session.execute(
        select(PricesDaily.symbol, name_col.label("name"))
        .outerjoin(Instrument, and_(Instrument.yahoo_symbol == PricesDaily.symbol, Instrument.yahoo_symbol.isnot(None)))
        .distinct()
        .order_by(name_col)
    )

    instruments = [{"symbol": row.symbol, "name": row.name} for row in result.all()]

    return {"instruments": instruments}


def _compute_form13f_change(
    shares: int,
    shares_prev: int | None,
    value: int = 0,
    value_prev: int | None = None,
) -> str:
    """
    Compute the quarter-over-quarter change label for a 13F position.

    Uses value-based % when both values are available (split-adjusted).
    Falls back to share-based % when value data is missing.

    Returns: "—" (no prior data), "New", "Closed", or "+X.X%".
    """
    if shares_prev is None:
        return "—"
    if shares == 0:
        return "Closed"
    if shares_prev == 0:
        return "New"
    pct = _safe_pct(value, value_prev, shares, shares_prev)
    return f"{pct:+.1f}%"


def _safe_pct(value: int, value_prev: int | None, shares: int, shares_prev: int) -> float:
    """
    Compute QoQ % change using value-based calculation where possible.

    Value-based is preferred (split-adjusted), but DB unit inconsistencies can occur
    (e.g. one quarter stored in thousands, another in dollars).  When the share-based
    direction contradicts value-based by a large margin we fall back to share-based.
    """
    pct_shares = (shares - shares_prev) / shares_prev * 100
    if not (value and value_prev):
        return pct_shares
    pct_value = (value - value_prev) / value_prev * 100
    # Sanity-check: if both agree on direction, trust value-based (split-adjusted).
    # If they disagree on sign (unit mismatch in DB), trust share-based.
    if (pct_value >= 0) == (pct_shares >= 0):
        return pct_value
    return pct_shares


def _compute_form13f_signal_score(
    shares: int,
    shares_prev: int | None,
    value: int = 0,
    value_prev: int | None = None,
) -> int:
    """
    Compute per-holder 13F signal score (-2 to +2).

    Uses value-based % change when both values are available and directionally consistent
    (split-adjusted). Falls back to share-based % on DB unit inconsistency.

    Score rules (contiguous, no gaps):
      +2: New (value >= MIN); or increase ≥1000% with value >= MIN
      +1: Increase 10% to 999%; or increase ≥1000% with value < MIN
       0: Stable (-30% to +10%); no prior data; or tiny New (value < MIN)
      -1: Trimmed (-90% to -30%)
      -2: Closed; or effective liquidation (≤-90%)
    """
    value = value or 0
    if shares_prev is None:
        return 0
    if shares == 0:
        return -2
    if shares_prev == 0:
        return 2 if value >= FORM13F_MIN_VALUE_NEW else 0
    pct = _safe_pct(value, value_prev, shares, shares_prev)
    if pct >= FORM13F_INCREASE_EFFECTIVE_NEW:
        return 2 if value >= FORM13F_MIN_VALUE_NEW else 1
    if pct <= FORM13F_TRIM_EFFECTIVE_LIQUIDATION:
        return -2
    if pct >= 10:
        return 1
    if pct < -30:
        return -1
    return 0


def _score_reason(score: int, change: str, value: int) -> str | None:
    """Human-readable explanation for why a holder has score=0 (shown in tooltip)."""
    if score != 0:
        return None  # contributing to score — no explanation needed
    if change == "—":
        return "no prior quarter for comparison"
    if change == "New" and value < FORM13F_MIN_VALUE_NEW:
        return f"new position but value below ${FORM13F_MIN_VALUE_NEW:,} minimum"
    if change not in ("New", "Closed"):
        return "change within stable range (−30% to +10%)"
    return None


async def _get_form13f_for_instruments(
    session: AsyncSession, instrument_ids: list[int]
) -> dict[int, Form13FInstrumentResult]:
    """
    Get 13F score and holders for each instrument.

    Score aggregation:
    1. Per-holder score: -2 to +2 from raw share counts (see _compute_form13f_signal_score)
    2. Exclude score=0 holders (no prior data or stable — no directional signal)
    3. Conviction-weighted average: weight = value / filing_total_value (treats funds equally by commitment %)
    4. Clamp to [-2, 2], round to 1 decimal

    Returns {instrument_id: {score, holders}}.
    """
    if not instrument_ids:
        return {}

    rows = (
        await session.execute(
            select(Form13FHolding, Form13FFiling, Form13FManager)
            .join(Form13FFiling, Form13FHolding.filing_id == Form13FFiling.id)
            .join(Form13FManager, Form13FFiling.manager_id == Form13FManager.id)
            .where(Form13FHolding.instrument_id.in_(instrument_ids))
        )
    ).all()

    # Also fetch the two most recent filing dates per manager so we can detect
    # genuinely new positions (manager had a prior filing but no prior holding).
    all_manager_ids = {manager.id for _, _, manager in rows}
    manager_filing_dates: dict[int, list[date]] = defaultdict(list)
    if all_manager_ids:
        filing_rows = (
            await session.execute(
                select(Form13FFiling.manager_id, Form13FFiling.report_date)
                .where(Form13FFiling.manager_id.in_(all_manager_ids))
                .order_by(Form13FFiling.manager_id, Form13FFiling.report_date.desc())
            )
        ).all()
        for mid, rdate in filing_rows:
            manager_filing_dates[mid].append(rdate)

    by_manager_filing: dict[tuple[int, int, int], dict[str, str | int | date | None]] = {}
    for holding, filing, manager in rows:
        key = (holding.instrument_id, manager.id, filing.id)
        if key not in by_manager_filing:
            by_manager_filing[key] = {
                "instrument_id": holding.instrument_id,
                "manager_name": manager.name,
                "manager_id": manager.id,
                "report_date": filing.report_date,
                "filing_total_value": filing.total_value,
                "value": 0,
                "shares": 0,
            }
        by_manager_filing[key]["value"] += holding.value
        by_manager_filing[key]["shares"] += holding.shares

    by_manager: dict[tuple[int, int], list[dict[str, str | int | date | None]]] = defaultdict(list)
    for (iid, mid, fid), data in by_manager_filing.items():
        by_manager[(iid, mid)].append(data)

    by_instrument: dict[int, list[dict[str, str | int | date | None]]] = defaultdict(list)
    for (iid, mid), filings_list in by_manager.items():
        filings_list.sort(key=lambda x: x["report_date"], reverse=True)
        latest = filings_list[0]
        prev = filings_list[1] if len(filings_list) > 1 else None

        if prev is not None:
            # Manager held this instrument in both quarters
            shares_prev: int | None = prev["shares"]
            value_prev: int | None = prev["value"]
        elif len(manager_filing_dates.get(mid, [])) >= 2:
            # Manager has a prior filing but didn't hold this instrument → new position
            shares_prev = 0
            value_prev = 0
        else:
            # Manager has only one filing total → no comparison possible
            shares_prev = None
            value_prev = None

        shares = latest["shares"]
        change = _compute_form13f_change(
            shares, shares_prev, value=latest["value"], value_prev=value_prev
        )
        score = _compute_form13f_signal_score(
            shares, shares_prev, value=latest["value"], value_prev=value_prev
        )
        filing_total = latest["filing_total_value"] or 0
        conviction = latest["value"] / filing_total if filing_total > 0 else 0.0
        by_instrument[iid].append({
            "manager_id": mid,
            "manager_name": latest["manager_name"],
            "change": change,
            "score": score,
            "value": latest["value"],
            "conviction": conviction,
            "report_date": latest["report_date"].isoformat() if latest.get("report_date") else None,
            "shares": latest["shares"],
            "shares_prev": shares_prev,
        })

    result: dict[int, Form13FInstrumentResult] = {}
    for iid, holders in by_instrument.items():
        # Only holders with a directional signal contribute to the score.
        # score == 0 means "no prior data" or "stable" — no useful information.
        scoring_holders = [h for h in holders if h["score"] != 0]

        if scoring_holders:
            # Weight by conviction (pct of manager's portfolio), not absolute dollars.
            # This treats a small fund putting 10% into a stock equally to a large fund doing the same.
            total_conviction = sum(h["conviction"] for h in scoring_holders)
            if total_conviction > 0:
                weighted_score = sum(h["score"] * h["conviction"] for h in scoring_holders) / total_conviction
            else:
                weighted_score = sum(h["score"] for h in scoring_holders) / len(scoring_holders)
        else:
            weighted_score = 0.0

        # Round only here (backend). Frontend displays as received.
        scoring_set = set(id(h) for h in scoring_holders)
        result[iid] = {
            "score": round(max(-2.0, min(2.0, weighted_score)), 1),
            "holders": [
                {
                    "manager_id": h.get("manager_id"),
                    "name": h["manager_name"],
                    "change": h["change"],
                    "report_date": h.get("report_date"),
                    "shares": h.get("shares"),
                    "shares_prev": h.get("shares_prev"),
                    "value": h.get("value"),
                    "scored": id(h) in scoring_set,
                    "score_reason": _score_reason(h["score"], h["change"], h.get("value") or 0),
                }
                for h in holders
            ],
        }
    return result


def _build_sec_13f_url(cik: Optional[str], accession: Optional[str]) -> Optional[str]:
    """Build SEC EDGAR URL for a 13F filing. Returns None if cik or accession missing."""
    if not cik or not accession:
        return None
    cik_num = str(cik).lstrip("0") or "0"
    accession_clean = str(accession).replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{accession_clean}/"


@app.get("/api/instrument/{symbol}")
async def get_instrument(
    symbol: str, days: int = 30, session: AsyncSession = Depends(get_db_session)
) -> dict[str, Any]:
    """Get detailed data for a specific stock by Yahoo symbol"""
    end_date = datetime.now(TIMEZONE).date()
    start_date = end_date - timedelta(days=days)

    instrument_result = await session.execute(
        select(Instrument)
        .filter(Instrument.yahoo_symbol == symbol)
        .options(selectinload(Instrument.yahoo), selectinload(Instrument.earnings_reports))
    )
    instrument = instrument_result.scalars().first()
    if not instrument:
        raise HTTPException(status_code=404, detail="Instrument not found")

    # Get price data for the period
    prices_result = await session.execute(
        select(PricesDaily)
        .where(PricesDaily.symbol == symbol, PricesDaily.date >= start_date)
        .order_by(PricesDaily.date.asc())
    )

    chart_price_data: dict[str, float] = {}
    prices = prices_result.scalars().all()
    for price in prices:
        chart_price_data[price.date.isoformat()] = getattr(price, PRICE_FIELD.lower().replace(" ", "_") + "_price")

    chart_orders_data: dict[str, dict[str, float | str]] = {}
    orders_result = await session.execute(
        select(TransactionHistory)
        .where(TransactionHistory.ticker == symbol, TransactionHistory.timestamp >= start_date)
        .order_by(TransactionHistory.timestamp)
    )
    orders = orders_result.scalars().all()
    for order in orders:
        chart_orders_data[order.timestamp.isoformat()] = {
            "action": order.action.value,
            "total": order.total,
        }

    yd = (instrument.yahoo.info or {}) if instrument.yahoo else {}

    fundamentals = {
        "marketCap": yd.get("marketCap"),
        "peRatio": yd.get("trailingPE"),
        "pegRatio": yd.get("trailingPegRatio"),
        "beta": yd.get("beta"),
        "dividendYield": yd.get("dividendYield"),
        "totalDebt": yd.get("totalDebt"),
        "totalCash": yd.get("totalCash"),
        "sharesOutstanding": yd.get("sharesOutstanding") or yd.get("impliedSharesOutstanding"),
        "freeCashflow": yd.get("freeCashflow"),
        "operatingCashflow": yd.get("operatingCashflow"),
        "totalRevenue": yd.get("totalRevenue"),
        "revenuePerShare": yd.get("revenuePerShare"),
        "revenueGrowth": yd.get("revenueGrowth"),
        "profitMargins": yd.get("profitMargins"),
        "returnOnAssets": yd.get("returnOnAssets"),
        "returnOnEquity": yd.get("returnOnEquity"),
        # Additional valuation fields
        "enterpriseValue": yd.get("enterpriseValue"),
        "enterpriseToEbitda": yd.get("enterpriseToEbitda"),
        "enterpriseToRevenue": yd.get("enterpriseToRevenue"),
        "priceToSalesTtm": yd.get("priceToSalesTrailing12Months"),
        "priceToBook": yd.get("priceToBook"),
        "ebitda": yd.get("ebitda"),
        "recommendationMean": yd.get("recommendationMean"),
        "recommendationKey": yd.get("recommendationKey"),
        "numberOfAnalystOpinions": yd.get("numberOfAnalystOpinions"),
        "fiftyTwoWeekHighChangePercent": yd.get("fiftyTwoWeekHighChangePercent"),
        "fiftyTwoWeekHigh": yd.get("fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow": yd.get("fiftyTwoWeekLow"),
        "_rawCurrency": yd.get("financialCurrency"),
    }

    # 13F institutional holders: match by instrument_id (set from Form13FHolding.cusip vs Instrument.isin)
    form13f_holdings: list[dict[str, str | int | float | None]] = []
    holdings_result = await session.execute(
        select(Form13FHolding, Form13FFiling, Form13FManager)
        .join(Form13FFiling, Form13FHolding.filing_id == Form13FFiling.id)
        .join(Form13FManager, Form13FFiling.manager_id == Form13FManager.id)
        .where(Form13FHolding.instrument_id == instrument.id)
    )
    rows = holdings_result.all()

    # Aggregate by (manager_id, filing_id): sum value, sum shares
    by_manager_filing: dict[tuple[int, int], Form13FFilingRow] = {}
    for holding, filing, manager in rows:
        key = (manager.id, filing.id)
        if key not in by_manager_filing:
            by_manager_filing[key] = {
                "manager_name": manager.name,
                "manager_id": manager.id,
                "manager_cik": manager.cik,
                "report_date": filing.report_date,
                "accession_number": filing.accession_number,
                "value": 0,
                "shares": 0,
                "filing_total_value": filing.total_value,
            }
        by_manager_filing[key]["value"] += holding.value
        by_manager_filing[key]["shares"] += holding.shares

    # Per manager: latest and prev filing, compute change
    by_manager: dict[int, list[Form13FFilingRow]] = {}
    for (mid, fid), data in by_manager_filing.items():
        by_manager.setdefault(mid, []).append(data)

    for mid, filings_list in by_manager.items():
        filings_list.sort(key=lambda x: x["report_date"], reverse=True)
        latest = filings_list[0]
        prev = filings_list[1] if len(filings_list) > 1 else None

        shares_prev = prev["shares"] if prev else None
        value_prev_h = prev["value"] if prev else None
        change = _compute_form13f_change(
            latest["shares"], shares_prev, value=latest["value"], value_prev=value_prev_h
        )
        report_date_prev = prev["report_date"].isoformat() if prev else None

        filing_total = latest.get("filing_total_value") or 0
        pct_of_portfolio = (
            (latest["value"] / filing_total * 100) if filing_total and filing_total > 0 else None
        )

        form13f_holdings.append({
            "manager_name": latest["manager_name"],
            "value": latest["value"],
            "pct_of_portfolio": round(pct_of_portfolio, 2) if pct_of_portfolio is not None else None,
            "shares": latest["shares"],
            "report_date": latest["report_date"].isoformat(),
            "change": change,
            "shares_prev": shares_prev,
            "report_date_prev": report_date_prev,
            "sec_filing_url": _build_sec_13f_url(latest.get("manager_cik"), latest.get("accession_number")),
        })

    # Sort by value descending (biggest holders first)
    form13f_holdings.sort(key=lambda x: x["value"], reverse=True)

    form13f_as_of = max((h["report_date"] for h in form13f_holdings), default=None)

    # User's position (if held): portfolio_pct, market_value, profit, return_pct
    my_position: Optional[dict[str, Any]] = None
    latest_date_result = await session.execute(select(func.max(HoldingDaily.date)))
    latest_date = latest_date_result.scalar()
    if latest_date:
        holding_result = await session.execute(
            select(HoldingDaily)
            .where(
                HoldingDaily.instrument_id == instrument.id,
                HoldingDaily.date == latest_date,
            )
            .options(selectinload(HoldingDaily.instrument))
        )
        user_holding = holding_result.scalar_one_or_none()
        if user_holding and user_holding.quantity > 0:
            currency_rates = await get_rates(session)
            all_holdings_result = await session.execute(
                select(HoldingDaily)
                .join(Instrument)
                .where(HoldingDaily.date == latest_date)
                .options(selectinload(HoldingDaily.instrument))
            )
            all_holdings = all_holdings_result.scalars().all()
            total_portfolio_value = sum(
                h.quantity * h.current_price * currency_rates.get(h.instrument.currency, 1.0)
                for h in all_holdings
            )
            market_value_gbp = (
                user_holding.quantity
                * user_holding.current_price
                * currency_rates.get(instrument.currency, 1.0)
            )
            portfolio_pct = (
                (market_value_gbp / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
            )
            profit = user_holding.ppl if user_holding.ppl is not None else 0
            cost_basis = (market_value_gbp - user_holding.ppl) if user_holding.ppl is not None else 0
            return_pct = (
                (user_holding.ppl / cost_basis * 100.0)
                if user_holding.ppl is not None and cost_basis > 0
                else 0.0
            )
            my_position = {
                "portfolio_pct": round(portfolio_pct, 2),
                "market_value": round(market_value_gbp, 2),
                "profit": round(profit, 2),
                "return_pct": round(return_pct, 2),
            }

    yh = instrument.yahoo
    return {
        "instrument": {
            "id": instrument.id,
            "symbol": instrument.yahoo_symbol,
            "t212_code": instrument.t212_code,
            "name": instrument.name,
            "currency": instrument.currency,
            "sector": yd.get("sector"),
            "country": yd.get("country"),
            "business_summary": yd.get("longBusinessSummary"),
            "quote_type": yd.get("quoteType"),
        },
        "fundamentals": fundamentals,
        "earnings": (yh.earnings or {}) if yh else {},
        "cashflow": (yh.cashflow or {}) if yh else {},
        "prices": chart_price_data,
        "orders": chart_orders_data,
        "pe_history": {
            k: v["pe_ratio"] for k, v in (yh.pes or {}).items() if date.fromisoformat(k) >= start_date
        } if yh else {},
        "splits": {k: v for k, v in (yh.splits or {}).items() if date.fromisoformat(k) >= start_date} if yh else {},
        "recommendations": (yh.recommendations or {}) if yh else {},
        "news": yh.news if yh else [],
        "earnings_reports": [
            {
                "id": report.id,
                "date": report.date.isoformat(),
                "summary": report.summary,
                "metrics": report.metrics,
                "created_at": report.created_at.isoformat() if report.created_at else None,
            }
            for report in sorted(instrument.earnings_reports, key=lambda x: x.date, reverse=True)
        ],
        "form13f_holdings": form13f_holdings,
        "form13f_as_of": form13f_as_of,
        "my_position": my_position,
    }


@app.get("/api/chart/prices")
async def get_chart_prices(
    symbols: str, days: int = 30, session: AsyncSession = Depends(get_db_session)
) -> dict[str, Any]:
    """Get daily price data for charting."""
    return await get_chart_metric(symbols, days, "price", session)


@app.get("/api/chart/metrics")
async def get_chart_metrics(
    symbols: str, days: int = 30, metric: str = "price", session: AsyncSession = Depends(get_db_session)
) -> dict[str, Any]:
    """Get chart data for different metrics."""
    return await get_chart_metric(symbols, days, metric, session)


@app.get("/api/screeners")
async def get_available_screeners() -> dict[str, Any]:
    """Get list of all available screeners with their configurations."""
    screener_config = get_screener_config()
    return screener_config.to_dict()


async def get_chart_metric(symbols: str, days: int, metric: str, session: AsyncSession) -> dict[str, Any]:
    """Get chart data for a specific metric."""
    if not symbols:
        raise HTTPException(status_code=400, detail="No symbols provided")

    # Parse symbols from comma-separated or space-separated string
    symbol_list = [s.strip().upper() for s in symbols.replace(",", " ").split() if s.strip()]

    # Calculate date range
    start_date = datetime.now(TIMEZONE).date() - timedelta(days=days)

    if metric == "price":
        # Get price data from database
        result = await session.execute(
            select(
                PricesDaily.symbol,
                PricesDaily.date,
                PRICE_COLUMN,
            ).filter(PricesDaily.symbol.in_(symbol_list), PricesDaily.date >= start_date)
        )
        price_data = result.all()

        # Convert to chart format
        chart_data: dict[str, list[dict[str, str | float]]] = defaultdict(list)
        for row in price_data:
            chart_data[row.symbol].append({"date": row.date.isoformat(), "value": row.price})
    else:
        # Get holdings data for other metrics
        # For valuation metrics, read from InstrumentMetricsDaily; for PnL metrics, read from holdings
        chart_data = defaultdict(list)
        if metric in {"pe_ratio", "institutional"}:
            result = await session.execute(
                select(InstrumentMetricsDaily, Instrument.yahoo_symbol)
                .join(Instrument, Instrument.id == InstrumentMetricsDaily.instrument_id)
                .where(
                    Instrument.yahoo_symbol.in_(symbol_list),
                    InstrumentMetricsDaily.date >= start_date,
                )
                .order_by(InstrumentMetricsDaily.date)
            )
            rows = result.all()
            for metrics, symbol in rows:
                value = None
                if metric == "pe_ratio":
                    value = metrics.pe_ratio
                elif metric == "institutional":
                    value = (metrics.institutional * 100) if metrics.institutional is not None else None
                if value is not None:
                    chart_data[symbol].append({"date": metrics.date.isoformat(), "value": value})
        else:
            result = await session.execute(
                select(HoldingDaily)
                .join(Instrument)
                .filter(
                    Instrument.yahoo_symbol.in_(symbol_list),
                    HoldingDaily.date >= start_date,
                )
                .order_by(HoldingDaily.date)
                .options(selectinload(HoldingDaily.instrument))
            )
            holdings_data = result.scalars().all()
            for holding in holdings_data:
                symbol = holding.instrument.yahoo_symbol
                if metric == "profit":
                    value = holding.ppl
                elif metric == "profit_pct":
                    market_value = holding.quantity * holding.current_price
                    value = (
                        round((holding.ppl / (market_value - holding.ppl) * 100.0), 2)
                        if (market_value - holding.ppl) > 0
                        else 0.0
                    )
                else:
                    value = None

                if value is not None:
                    chart_data[symbol].append({"date": holding.date.isoformat(), "value": float(value)})

        # Sort series by date
        for sym in chart_data:
            chart_data[sym].sort(key=lambda x: x["date"])

    return {
        "symbols": symbol_list,
        "data": chart_data,
        "days": days,
        "metric": metric,
    }


@app.get("/api/market/movers")
async def get_movers(
    period: str = "1d",
    session: AsyncSession = Depends(get_db_session),
) -> list[dict[str, Any]]:
    """Get all portfolio movers (holdings with price changes) for a given period."""
    valid_periods = {"1d": 1, "1w": 7, "1m": 30, "90d": 90}
    if period not in valid_periods:
        raise HTTPException(status_code=400, detail="Invalid period. Use: 1d, 1w, 1m, 90d")

    days = valid_periods[period]
    today = datetime.now(TIMEZONE).date()
    start_date = today - timedelta(days=days)

    currency_rates = await get_rates(session)

    result = await session.execute(
        select(
            Instrument.yahoo_symbol,
            Instrument.name,
            Instrument.t212_code,
            Instrument.currency,
            HoldingDaily.date,
            HoldingDaily.current_price,
            HoldingDaily.ppl,
            HoldingDaily.quantity,
        )
        .join(HoldingDaily)
        .filter(HoldingDaily.date >= start_date)
        .order_by(Instrument.yahoo_symbol, HoldingDaily.date)
    )
    instruments = result.all()

    prices_data = defaultdict(list)

    for price in instruments:
        prices_data[price.yahoo_symbol].append(price)

    # Calculate percentage changes
    movers = []
    for symbol, symbol_data in prices_data.items():
        if len(symbol_data) >= 2:
            # Get first and last prices
            first_price = symbol_data[0].current_price
            last_price = symbol_data[-1].current_price

            if first_price > 0:
                change_pct = ((last_price - first_price) / first_price) * 100

                market_value_gbp = (
                    symbol_data[-1].quantity * symbol_data[-1].current_price * currency_rates[symbol_data[-1].currency]
                )
                gain_pct = symbol_data[-1].ppl / (market_value_gbp - symbol_data[-1].ppl) * 100.0

                movers.append(
                    {
                        "symbol": symbol,
                        "name": symbol_data[0].name,
                        "change_pct": change_pct,
                        "current_price": last_price,
                        "t212_code": symbol_data[0].t212_code,
                        "gain_pct": gain_pct,
                        "value": market_value_gbp,
                    }
                )

    # Sort by percentage change (descending: highest gainers first)
    movers.sort(key=lambda x: x["change_pct"], reverse=True)

    return movers


@app.get("/api/pies")
async def get_pies(session: AsyncSession = Depends(get_db_session)):
    """Get all pies with their instruments."""
    # Fetch all pies with their instruments and related instrument data
    result = await session.execute(
        select(Pie).options(selectinload(Pie.instruments).selectinload(PieInstrument.instrument)).order_by(Pie.name)
    )
    pies = result.scalars().all()

    # Format the response
    pies_data = [
        {
            "id": pie.id,
            "name": pie.name,
            "cash": pie.cash,
            "progress": pie.progress,
            "status": pie.status,
            "creation_date": pie.creation_date.isoformat() if pie.creation_date else None,
            "end_date": pie.end_date.isoformat() if pie.end_date else None,
            "dividend_cash_action": pie.dividend_cash_action,
            "goal": pie.goal,
            "dividend_details": pie.dividend_details,
            "result": pie.result,
            "instruments": [
                {
                    "t212_code": instrument.t212_code,
                    "instrument_name": instrument.instrument.name if instrument.instrument else None,
                    "yahoo_symbol": instrument.instrument.yahoo_symbol if instrument.instrument else None,
                    "expected_share": instrument.expected_share,
                    "current_share": instrument.current_share,
                    "owned_quantity": instrument.owned_quantity,
                    "result": instrument.result,
                    "issues": instrument.issues,
                }
                for instrument in sorted(pie.instruments, key=lambda i: i.current_share, reverse=True)
            ],
        }
        for pie in pies
    ]

    return pies_data


def _aggregate_holdings_by_cusip(holdings: list) -> dict[str, dict]:
    """Aggregate Form13FHolding rows by CUSIP (sums value+shares across share classes)."""
    by_cusip: dict[str, dict] = {}
    for h in holdings:
        if h.cusip not in by_cusip:
            by_cusip[h.cusip] = {
                "cusip": h.cusip,
                "issuer": h.issuer,
                "value": 0,
                "shares": 0,
                "instrument_id": h.instrument_id,
            }
        else:
            # Prefer the first non-None instrument_id we find for this CUSIP
            if by_cusip[h.cusip]["instrument_id"] is None and h.instrument_id is not None:
                by_cusip[h.cusip]["instrument_id"] = h.instrument_id
        by_cusip[h.cusip]["value"] += h.value
        by_cusip[h.cusip]["shares"] += h.shares
    return by_cusip


def _change_sort_value(
    shares: int,
    shares_prev: Optional[int],
    value: int = 0,
    value_prev: Optional[int] = None,
) -> float:
    """Numeric sort key for QoQ change: New=+1e9, Closed=-1e9, else value-based %."""
    if shares_prev is None:
        return 0.0
    if shares == 0:
        return -1.0e9
    if shares_prev == 0:
        return 1.0e9
    return _safe_pct(value, value_prev, shares, shares_prev)


def _build_portfolio_and_moves(
    latest_by_cusip: dict,
    prev_by_cusip: dict,
    total_value: int,
    instruments: dict[int, Any],
) -> tuple[list[dict], dict]:
    """
    Compute per-position portfolio rows and the moves summary (new/closed/buys/sells).
    `instruments` is a dict of {instrument_id: Instrument}.
    Returns (portfolio_rows sorted by value desc, moves dict).
    """
    portfolio: list[dict] = []
    for cusip, item in latest_by_cusip.items():
        prev = prev_by_cusip.get(cusip)
        if prev is not None:
            shares_prev: Optional[int] = prev["shares"]
            value_prev: Optional[int] = prev["value"]
        elif prev_by_cusip:
            # Previous filing exists but this CUSIP wasn't in it → new position
            shares_prev = 0
            value_prev = 0
        else:
            shares_prev = None
            value_prev = None

        value_change = (item["value"] - value_prev) if value_prev is not None else None
        change = _compute_form13f_change(
            item["shares"], shares_prev, value=item["value"], value_prev=value_prev
        )

        instr = instruments.get(item["instrument_id"])
        pct = item["value"] / total_value * 100 if total_value > 0 else 0.0

        portfolio.append({
            "cusip": cusip,
            "issuer": item["issuer"],
            "yahoo_symbol": instr.yahoo_symbol if instr else None,
            "name": instr.name if instr else item["issuer"],
            "value": item["value"],
            "shares": item["shares"],
            "pct_of_portfolio": round(pct, 2),
            "change": change,
            "change_sort": _change_sort_value(
                item["shares"], shares_prev, value=item["value"], value_prev=value_prev
            ),
            "value_change": value_change,
            "value_prev": value_prev,
            "shares_prev": shares_prev,
        })

    portfolio.sort(key=lambda x: x["value"], reverse=True)
    for i, p in enumerate(portfolio):
        p["rank"] = i + 1

    # Closed positions: in prev but missing from latest
    closed: list[dict] = []
    for cusip, prev in prev_by_cusip.items():
        if cusip not in latest_by_cusip:
            instr = instruments.get(prev["instrument_id"])
            closed.append({
                "cusip": cusip,
                "issuer": prev["issuer"],
                "yahoo_symbol": instr.yahoo_symbol if instr else None,
                "name": instr.name if instr else prev["issuer"],
                "value": 0,
                "value_prev": prev["value"],
                "shares": 0,
                "shares_prev": prev["shares"],
                "change": "Closed",
                "change_sort": -1.0e9,
                "value_change": -prev["value"],
            })
    closed.sort(key=lambda x: (x["value_prev"] or 0), reverse=True)

    new_positions = sorted(
        [p for p in portfolio if p["change"] == "New"],
        key=lambda x: x["value"], reverse=True,
    )
    existing = [p for p in portfolio if p["change"] not in ("New", "Closed", "—")]
    all_buys = sorted(
        [p for p in existing if (p["value_change"] or 0) > 0],
        key=lambda x: x["value_change"], reverse=True,
    )
    all_sells = sorted(
        [p for p in existing if (p["value_change"] or 0) < 0],
        key=lambda x: x["value_change"],
    )

    moves = {
        "new_positions": new_positions,
        "closed_positions": closed,
        "top_buys": all_buys[:10],
        "top_sells": all_sells[:10],
        # Actual counts for stats bar (not capped at 10 like the display lists)
        "increased_count": len(all_buys),
        "trimmed_count": len(all_sells),
    }
    return portfolio, moves


@app.get("/api/13f/highlights")
async def get_form13f_highlights(session: AsyncSession = Depends(get_db_session)) -> dict:
    """
    Cross-manager consensus signals from the latest 13F filings.
    Returns the stocks being bought / sold by the most managers, and the most widely held.
    """
    managers_result = await session.execute(select(Form13FManager).order_by(Form13FManager.name))
    managers = managers_result.scalars().all()
    if not managers:
        return {"most_bought": [], "most_sold": [], "most_held": []}

    manager_ids = [m.id for m in managers]
    filings_result = await session.execute(
        select(Form13FFiling)
        .where(Form13FFiling.manager_id.in_(manager_ids))
        .order_by(Form13FFiling.manager_id, Form13FFiling.report_date.desc())
    )
    all_filings = filings_result.scalars().all()

    filings_by_manager: dict[int, list] = defaultdict(list)
    for f in all_filings:
        filings_by_manager[f.manager_id].append(f)

    latest_ids: dict[int, int] = {}
    prev_ids: dict[int, int] = {}
    for mid, mf in filings_by_manager.items():
        latest_ids[mid] = mf[0].id
        if len(mf) >= 2:
            prev_ids[mid] = mf[1].id

    relevant_ids = list(set(latest_ids.values()) | set(prev_ids.values()))
    holdings_result = await session.execute(
        select(Form13FHolding).where(Form13FHolding.filing_id.in_(relevant_ids))
    )
    all_holdings = holdings_result.scalars().all()

    instrument_ids = {h.instrument_id for h in all_holdings if h.instrument_id is not None}
    instruments: dict[int, Any] = {}
    if instrument_ids:
        instr_result = await session.execute(select(Instrument).where(Instrument.id.in_(instrument_ids)))
        for instr in instr_result.scalars().all():
            instruments[instr.id] = instr

    holdings_by_filing: dict[int, list] = defaultdict(list)
    for h in all_holdings:
        holdings_by_filing[h.filing_id].append(h)

    manager_by_id = {m.id: m for m in managers}

    buying: dict[str, dict] = {}
    selling: dict[str, dict] = {}
    held: dict[str, dict] = {}

    for mid, mf in filings_by_manager.items():
        manager_name = manager_by_id[mid].name
        latest_by_cusip = _aggregate_holdings_by_cusip(holdings_by_filing.get(latest_ids[mid], []))
        prev_by_cusip = (
            _aggregate_holdings_by_cusip(holdings_by_filing.get(prev_ids[mid], []))
            if mid in prev_ids else {}
        )

        # Most held (count every manager that holds this stock currently)
        for cusip, item in latest_by_cusip.items():
            instr = instruments.get(item["instrument_id"])
            if cusip not in held:
                held[cusip] = {
                    "cusip": cusip,
                    "issuer": item["issuer"],
                    "yahoo_symbol": instr.yahoo_symbol if instr else None,
                    "name": instr.name if instr else item["issuer"],
                    "count": 0,
                    "managers": [],
                    "total_value": 0,
                    # buy_count / sell_count populated below when prev data is available
                    "buy_count": 0,
                    "sell_count": 0,
                }
            held[cusip]["count"] += 1
            held[cusip]["managers"].append(manager_name)
            held[cusip]["total_value"] += item["value"]

        if not prev_by_cusip:
            continue  # No prev quarter → can't compute moves for this manager

        # Buying signals: new or increased positions
        for cusip, item in latest_by_cusip.items():
            prev = prev_by_cusip.get(cusip)
            # prev_by_cusip is non-empty here (guarded above), so None → new position
            shares_prev = prev["shares"] if prev is not None else 0
            value_prev_buy = prev["value"] if prev is not None else 0
            score = _compute_form13f_signal_score(
                item["shares"], shares_prev, item["value"], value_prev=value_prev_buy
            )
            if score <= 0:
                continue
            change = _compute_form13f_change(
                item["shares"], shares_prev, value=item["value"], value_prev=value_prev_buy
            )
            value_added = item["value"] - value_prev_buy
            instr = instruments.get(item["instrument_id"])
            if cusip not in buying:
                buying[cusip] = {
                    "cusip": cusip,
                    "issuer": item["issuer"],
                    "yahoo_symbol": instr.yahoo_symbol if instr else None,
                    "name": instr.name if instr else item["issuer"],
                    "count": 0,
                    "managers": [],
                    "total_value_added": 0,
                }
            buying[cusip]["count"] += 1
            buying[cusip]["managers"].append({"name": manager_name, "change": change})
            buying[cusip]["total_value_added"] += value_added

        # Selling signals: closed or significantly trimmed positions
        for cusip, prev_item in prev_by_cusip.items():
            latest_item = latest_by_cusip.get(cusip)
            if latest_item is None:
                change = "Closed"
                value_removed = prev_item["value"]
                instr_id = prev_item["instrument_id"]
                issuer = prev_item["issuer"]
            else:
                score = _compute_form13f_signal_score(
                    latest_item["shares"], prev_item["shares"],
                    latest_item["value"], value_prev=prev_item["value"],
                )
                if score >= 0:
                    continue
                change = _compute_form13f_change(
                    latest_item["shares"], prev_item["shares"],
                    value=latest_item["value"], value_prev=prev_item["value"],
                )
                value_removed = prev_item["value"] - latest_item["value"]
                instr_id = latest_item["instrument_id"] or prev_item["instrument_id"]
                issuer = latest_item["issuer"]

            instr = instruments.get(instr_id)
            if cusip not in selling:
                selling[cusip] = {
                    "cusip": cusip,
                    "issuer": issuer,
                    "yahoo_symbol": instr.yahoo_symbol if instr else None,
                    "name": instr.name if instr else issuer,
                    "count": 0,
                    "managers": [],
                    "total_value_removed": 0,
                }
            selling[cusip]["count"] += 1
            selling[cusip]["managers"].append({"name": manager_name, "change": change})
            selling[cusip]["total_value_removed"] += value_removed

    # Merge buy/sell counts per CUSIP and compute net signal
    # Propagate buy/sell trend counts into held items for trend indicator
    for cusip, h in held.items():
        if cusip in buying:
            h["buy_count"] += buying[cusip]["count"]
        if cusip in selling:
            h["sell_count"] += selling[cusip]["count"]

    all_cusips = set(buying) | set(selling)
    consensus_buys: list[dict] = []
    consensus_sells: list[dict] = []
    disputed: list[dict] = []

    for cusip in all_cusips:
        buy = buying.get(cusip)
        sell = selling.get(cusip)
        buy_count = buy["count"] if buy else 0
        sell_count = sell["count"] if sell else 0
        net = buy_count - sell_count

        base = buy if buy else sell
        item: dict = {
            "cusip": cusip,
            "issuer": base["issuer"],
            "yahoo_symbol": base["yahoo_symbol"],
            "name": base["name"],
            "buy_count": buy_count,
            "sell_count": sell_count,
            "net_managers": net,
            "total_value_added": buy["total_value_added"] if buy else 0,
            "total_value_removed": sell["total_value_removed"] if sell else 0,
            "buy_managers": buy["managers"] if buy else [],
            "sell_managers": sell["managers"] if sell else [],
        }

        if buy_count > 0 and sell_count > 0 and net == 0:
            disputed.append(item)
        elif net > 0:
            consensus_buys.append(item)
        elif net < 0:
            consensus_sells.append(item)

    top_n = 8
    most_bought = sorted(
        consensus_buys,
        key=lambda x: (x["net_managers"], x["buy_count"], x["total_value_added"]),
        reverse=True,
    )[:top_n]
    most_sold = sorted(
        consensus_sells,
        key=lambda x: (-x["net_managers"], x["sell_count"], x["total_value_removed"]),
        reverse=True,
    )[:top_n]
    most_held = sorted(held.values(), key=lambda x: (x["count"], x["total_value"]), reverse=True)[:top_n]
    top_disputed = sorted(
        disputed,
        key=lambda x: (x["buy_count"] + x["sell_count"]),
        reverse=True,
    )[:5]

    return {
        "most_bought": most_bought,
        "most_sold": most_sold,
        "most_held": most_held,
        "disputed": top_disputed,
    }


@app.get("/api/13f/managers")
async def get_form13f_managers_list(session: AsyncSession = Depends(get_db_session)) -> list:
    """List all 13F managers with their latest filing summary and QoQ activity counts."""
    managers_result = await session.execute(
        select(Form13FManager).order_by(Form13FManager.name)
    )
    managers = managers_result.scalars().all()
    if not managers:
        return []

    manager_ids = [m.id for m in managers]

    filings_result = await session.execute(
        select(Form13FFiling)
        .where(Form13FFiling.manager_id.in_(manager_ids))
        .order_by(Form13FFiling.manager_id, Form13FFiling.report_date.desc())
    )
    all_filings = filings_result.scalars().all()

    # Group filings per manager (already ordered desc by date)
    filings_by_manager: dict[int, list] = defaultdict(list)
    for f in all_filings:
        filings_by_manager[f.manager_id].append(f)

    # Collect the latest + prev filing IDs to load holdings in one query
    latest_ids: dict[int, int] = {}  # manager_id → latest filing_id
    prev_ids: dict[int, int] = {}
    for mid, mf in filings_by_manager.items():
        latest_ids[mid] = mf[0].id
        if len(mf) >= 2:
            prev_ids[mid] = mf[1].id

    relevant_filing_ids = list(set(latest_ids.values()) | set(prev_ids.values()))
    holdings_result = await session.execute(
        select(Form13FHolding).where(Form13FHolding.filing_id.in_(relevant_filing_ids))
    )
    all_holdings = holdings_result.scalars().all()

    # Group holdings by filing_id
    holdings_by_filing: dict[int, list] = defaultdict(list)
    for h in all_holdings:
        holdings_by_filing[h.filing_id].append(h)

    manager_by_id = {m.id: m for m in managers}
    result = []
    for mid in manager_ids:
        m = manager_by_id[mid]
        mf = filings_by_manager.get(mid, [])
        if not mf:
            continue

        latest_filing = mf[0]
        latest_cusips = _aggregate_holdings_by_cusip(holdings_by_filing.get(latest_ids[mid], []))
        prev_cusips = _aggregate_holdings_by_cusip(holdings_by_filing.get(prev_ids.get(mid), [])) if mid in prev_ids else {}

        # Count activity types
        activity: dict[str, int] = {"new": 0, "closed": 0, "increased": 0, "trimmed": 0, "stable": 0}
        if prev_cusips:
            for cusip, item in latest_cusips.items():
                prev = prev_cusips.get(cusip)
                # prev_cusips is non-empty, so None here means the CUSIP wasn't held last quarter → new position
                shares_prev = prev["shares"] if prev is not None else 0
                value_prev_act = prev["value"] if prev is not None else 0
                change = _compute_form13f_change(
                    item["shares"], shares_prev, value=item["value"], value_prev=value_prev_act
                )
                if change == "New":
                    activity["new"] += 1
                elif change == "Closed":
                    activity["closed"] += 1
                elif change == "—":
                    activity["stable"] += 1
                else:
                    m_obj = change.replace("+", "").replace("%", "")
                    try:
                        pct = float(m_obj)
                        if pct >= 10:
                            activity["increased"] += 1
                        elif pct <= -30:
                            activity["trimmed"] += 1
                        else:
                            activity["stable"] += 1
                    except ValueError:
                        activity["stable"] += 1
            for cusip in prev_cusips:
                if cusip not in latest_cusips:
                    activity["closed"] += 1

        result.append({
            "id": m.id,
            "name": m.name,
            "cik": m.cik,
            "latest_report_date": latest_filing.report_date.isoformat(),
            "total_value": latest_filing.total_value,
            "num_positions": len(latest_cusips),
            "filing_count": len(mf),
            "activity": activity if prev_cusips else None,
        })

    return result


@app.get("/api/13f/managers/{manager_id}")
async def get_form13f_manager_detail(
    manager_id: int,
    report_date: Optional[str] = None,
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    """Full portfolio + moves for a single manager for a given quarter (default: latest)."""
    manager_result = await session.execute(
        select(Form13FManager).where(Form13FManager.id == manager_id)
    )
    manager = manager_result.scalar_one_or_none()
    if not manager:
        raise HTTPException(status_code=404, detail="Manager not found")

    filings_result = await session.execute(
        select(Form13FFiling)
        .where(Form13FFiling.manager_id == manager_id)
        .order_by(Form13FFiling.report_date.desc())
    )
    filings = filings_result.scalars().all()
    if not filings:
        raise HTTPException(status_code=404, detail="No filings found for this manager")

    available_quarters = [f.report_date.isoformat() for f in filings]

    # Resolve which filing to show
    target_filing = filings[0]
    if report_date:
        try:
            target_date = date.fromisoformat(report_date)
            target_filing = next((f for f in filings if f.report_date == target_date), filings[0])
        except ValueError:
            pass  # fall back to latest

    # Find the previous filing (the one just before the target)
    target_idx = filings.index(target_filing)
    prev_filing = filings[target_idx + 1] if target_idx + 1 < len(filings) else None

    filing_ids = [target_filing.id] + ([prev_filing.id] if prev_filing else [])
    holdings_result = await session.execute(
        select(Form13FHolding).where(Form13FHolding.filing_id.in_(filing_ids))
    )
    all_holdings = holdings_result.scalars().all()

    # Load instruments referenced by any of these holdings
    instrument_ids = {h.instrument_id for h in all_holdings if h.instrument_id is not None}
    instruments: dict[int, Any] = {}
    if instrument_ids:
        instr_result = await session.execute(
            select(Instrument).where(Instrument.id.in_(instrument_ids))
        )
        for instr in instr_result.scalars().all():
            instruments[instr.id] = instr

    holdings_by_filing: dict[int, list] = defaultdict(list)
    for h in all_holdings:
        holdings_by_filing[h.filing_id].append(h)

    latest_by_cusip = _aggregate_holdings_by_cusip(holdings_by_filing[target_filing.id])
    prev_by_cusip = _aggregate_holdings_by_cusip(holdings_by_filing[prev_filing.id]) if prev_filing else {}

    portfolio, moves = _build_portfolio_and_moves(
        latest_by_cusip, prev_by_cusip, target_filing.total_value, instruments
    )

    cik_num = str(manager.cik).lstrip("0") or "0"
    sec_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik_num}&type=13F-HR&dateb=&owner=include&count=10"

    return {
        "manager": {"id": manager.id, "name": manager.name, "cik": manager.cik, "sec_url": sec_url},
        "available_quarters": available_quarters,
        "report_date": target_filing.report_date.isoformat(),
        "prev_report_date": prev_filing.report_date.isoformat() if prev_filing else None,
        "total_value": target_filing.total_value,
        "num_positions": len(portfolio),
        "portfolio": portfolio,
        "moves": moves,
    }


@app.get("/api/earnings-report/{symbol}/{report_date}", response_class=HTMLResponse)
async def get_earnings_report_html(symbol: str, report_date: str) -> HTMLResponse:
    """
    Get the full HTML earnings report file for a given symbol and date.
    For local dev server only - production should use nginx to serve static files directly.
    """
    project_root = Path(__file__).parent.parent
    filings_dir = project_root / "data" / "filings" / symbol

    if not filings_dir.exists():
        raise HTTPException(status_code=404, detail=f"Report directory not found for {symbol}")

    matching_files = sorted(filings_dir.glob(f"{report_date}_*.html"))

    if not matching_files:
        raise HTTPException(status_code=404, detail=f"Report file not found for {symbol} on {report_date}")

    html_content = matching_files[0].read_text(encoding="utf-8")

    return HTMLResponse(content=html_content, status_code=200, media_type="text/html")
