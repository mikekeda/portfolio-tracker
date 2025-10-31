"""
FastAPI backend for Trading212 Portfolio Manager
"""

# Standard library imports
import asyncio
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Optional

# Third-party imports
import aiohttp
import numpy as np

# Local imports
from backend.app import app, get_db_session
from dateutil.relativedelta import relativedelta
from fastapi import Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.screener_config import get_screener_config
from backend.utils.dcf import get_dcf_prices
from backend.utils.market_data import gen_buffett_indicator, gen_fear_greed_index, get_yield_spread
from backend.utils.roic import get_roic
from backend.utils.screener import calculate_screener_results
from backend.utils.technical import calculate_technical_indicators_for_symbols
from config import BENCHES, CURRENCIES, PRICE_FIELD, TIMEZONE, VIX
from data import QUICK_RATIO_THRESHOLDS
from models import (
    CurrencyRateDaily,
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
async def get_current_portfolio(session: AsyncSession = Depends(get_db_session)) -> dict[str, Any]:
    """Get current portfolio holdings with detailed information."""
    try:
        # Get the latest snapshot date
        result = await session.execute(select(func.max(HoldingDaily.date)))
        latest_date = result.scalar()

        if not latest_date:
            return {"holdings": [], "total_holdings": 0, "last_updated": None}

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

        # Fetch market metrics for latest_date in one batch
        instrument_ids = {h.instrument_id for h in holdings}
        metrics_result = await session.execute(
            select(InstrumentMetricsDaily).where(
                InstrumentMetricsDaily.date == latest_date,
                InstrumentMetricsDaily.instrument_id.in_(instrument_ids),
            )
        )
        metrics_rows = metrics_result.scalars().all()
        metrics_by_instrument_id = {m.instrument_id: m for m in metrics_rows}

        # Calculate technical indicators using centralized function
        symbols_for_technical = [h.instrument.yahoo_symbol for h in holdings if h.instrument.yahoo_symbol]
        rsi_data, technical_data = await calculate_technical_indicators_for_symbols(symbols_for_technical, session)
        dcf_prices = await get_dcf_prices([h.instrument for h in holdings])
        dcf_prices_dict = dict(zip(symbols_for_technical, dcf_prices))

        # Calculate total portfolio value for percentage calculation
        total_portfolio_value = 0.0
        for holding in holdings:
            market_value_native = holding.quantity * holding.current_price
            market_value_gbp = market_value_native * currency_rates[holding.instrument.currency]
            total_portfolio_value += market_value_gbp

        portfolio_data = []
        for holding in holdings:
            market_value_native = holding.quantity * holding.current_price
            market_value_gbp = market_value_native * currency_rates[holding.instrument.currency]
            portfolio_pct = (market_value_gbp / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
            dcf_price = dcf_prices_dict[holding.instrument.yahoo_symbol]

            # Yahoo Finance info for this instrument
            info = holding.instrument.yahoo.info
            metrics = metrics_by_instrument_id[holding.instrument_id]
            trends = calculate_historical_trends(holding)

            portfolio_data.append(
                {
                    "t212_code": holding.instrument.t212_code,
                    "name": holding.instrument.name,
                    "yahoo_symbol": holding.instrument.yahoo_symbol,
                    "currency": holding.instrument.currency,
                    "sector": holding.instrument.yahoo.info.get("sector"),
                    "country": holding.instrument.yahoo.info.get("country"),
                    "quantity": holding.quantity,
                    "avg_price": holding.avg_price,
                    "current_price": holding.current_price,
                    "analyst_price_targets": holding.instrument.yahoo.analyst_price_targets,
                    "dcf_price": dcf_price,
                    "dcf_diff": dcf_price / holding.current_price - 1 if dcf_price else None,
                    "ppl": holding.ppl,
                    "fx_ppl": holding.fx_ppl,
                    "market_cap": metrics.market_cap,
                    "pe_ratio": metrics.pe_ratio,
                    "ps_ratio": holding.instrument.yahoo.info.get("priceToSalesTrailing12Months"),
                    "avg_pe": holding.instrument.yahoo.avg_pe_5y,
                    "beta": metrics.beta,
                    "date": holding.date.isoformat(),
                    "market_value": market_value_gbp,  # Now in GBP
                    "profit": holding.ppl,  # Total profit (same as terminal - ppl already includes FX)
                    "return_pct": (holding.ppl / (market_value_gbp - holding.ppl) * 100.0)
                    if (market_value_gbp - holding.ppl) > 0
                    else 0.0,
                    "portfolio_pct": portfolio_pct,
                    "dividend_yield": info.get("dividendYield"),
                    "business_summary": info.get("longBusinessSummary"),
                    "prediction": (info["targetMedianPrice"] / holding.current_price - 1) * 100.0
                    if info.get("targetMedianPrice")
                    else None,
                    "institutional_ownership": round(metrics.institutional * 100.0)
                    if (metrics and metrics.institutional is not None)
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
                    if holding.instrument.yahoo.info.get("sector") != "Financial Services"
                    else None,
                    "debtToEquity": info.get("debtToEquity"),
                    "recommendation_mean": round(info["recommendationMean"], 2)
                    if info.get("recommendationMean")
                    else None,
                    "recommendation_key": info.get("recommendationKey"),
                    "recommendations": holding.instrument.yahoo.recommendations,
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
                    "rsi": rsi_data[holding.instrument.yahoo_symbol],
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
                    "bb_width_20_p30_6m": technical_data.get(holding.instrument.yahoo_symbol, {}).get(
                        "bb_width_20_p30_6m"
                    ),
                    "vol20_lt_vol60": technical_data.get(holding.instrument.yahoo_symbol, {}).get("vol20_lt_vol60"),
                    "volume_ratio": technical_data.get(holding.instrument.yahoo_symbol, {}).get("volume_ratio"),
                    **trends,
                    "quote_type": info.get("quoteType", "Unknown"),
                    "passedScreeners": [],  # will be populated below
                    "screener_score": 0,  # will be populated below
                }
            )

        # Calculate screener results for each holding
        calculate_screener_results(portfolio_data)

        return {
            "holdings": portfolio_data,
            "total_holdings": len(portfolio_data),
            "quick_ratio_thresholds": QUICK_RATIO_THRESHOLDS,
            "last_updated": latest_date.isoformat(),
        }
    except Exception as e:
        logger.error(f"Error fetching current portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/summary")
async def get_portfolio_summary(session: AsyncSession = Depends(get_db_session)) -> dict[str, Any]:
    """Get portfolio summary statistics including total value, profit, and win rate."""
    try:
        # Get the latest portfolio snapshot
        result = await session.execute(select(PortfolioDaily).order_by(PortfolioDaily.date.desc()))
        latest_snapshot = result.scalars().first()

        if not latest_snapshot:
            return {"error": "No portfolio data available"}

        # Get holdings for the same date to calculate win rate
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=False), raise_for_status=True
        ) as aiohttp_session:
            holdings_result, vix_result, fear_greed_index, yield_spread, buffett_indicator = await asyncio.gather(
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
            "last_updated": latest_snapshot.date.isoformat(),
            "vix": vix_result.scalar(),
            "fear_greed_index": fear_greed_index,
            "yield_spread": yield_spread,
            "buffett_indicator": buffett_indicator,
        }
    except Exception as e:
        logger.error(f"Error fetching portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/allocations")
async def get_portfolio_allocations(session: AsyncSession = Depends(get_db_session)) -> dict[str, Any]:
    """Get portfolio allocations by sector and country."""
    try:
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
    except Exception as e:
        logger.error(f"Error fetching allocations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/history")
async def get_portfolio_history(
    days: Optional[int] = None, session: AsyncSession = Depends(get_db_session)
) -> dict[str, Any]:
    """Get portfolio history for the last N days."""
    try:
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
    except Exception as e:
        logger.error(f"Error fetching portfolio history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/instruments")
async def get_instruments(session: AsyncSession = Depends(get_db_session)) -> dict[str, list[dict[str, Any]]]:
    """Get all instruments in the database for autocomplete."""
    try:
        result = await session.execute(
            select(Instrument).filter(Instrument.yahoo_symbol.isnot(None)).order_by(Instrument.name)
        )
        instruments = result.scalars().all()

        return {
            "instruments": [
                {
                    "id": instrument.id,
                    "symbol": instrument.yahoo_symbol,
                    "name": instrument.name,
                    "t212_code": instrument.t212_code,
                }
                for instrument in instruments
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching instruments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/instrument/{symbol}")
async def get_instrument(
    symbol: str, days: int = 30, session: AsyncSession = Depends(get_db_session)
) -> dict[str, Any]:
    """Get detailed data for a specific stock by Yahoo symbol"""
    try:
        end_date = datetime.now(TIMEZONE).date()
        start_date = end_date - timedelta(days=days)

        instrument_result = await session.execute(
            select(Instrument).filter(Instrument.yahoo_symbol == symbol).options(selectinload(Instrument.yahoo))
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

        yd = instrument.yahoo.info or {}

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
            "_rawCurrency": yd.get("financialCurrency"),
        }

        return {
            "instrument": {
                "id": instrument.id,
                "symbol": instrument.yahoo_symbol,
                "t212_code": instrument.t212_code,
                "name": instrument.name,
                "currency": instrument.currency,
                "sector": instrument.yahoo.info.get("sector"),
                "country": instrument.yahoo.info.get("country"),
                "business_summary": yd.get("longBusinessSummary"),
                "quote_type": yd.get("quoteType"),
            },
            "fundamentals": fundamentals,
            "earnings": instrument.yahoo.earnings or {},
            "cashflow": instrument.yahoo.cashflow or {},
            "prices": chart_price_data,
            "orders": chart_orders_data,
            "pe_history": {
                k: v["pe_ratio"] for k, v in instrument.yahoo.pes.items() if date.fromisoformat(k) >= start_date
            },
            "splits": {k: v for k, v in instrument.yahoo.splits.items() if date.fromisoformat(k) >= start_date},
            "recommendations": instrument.yahoo.recommendations or {},
            "news": instrument.yahoo.news,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching instrument {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    try:
        screener_config = get_screener_config()
        return screener_config.to_dict()
    except Exception as e:
        logger.error(f"Error getting available screeners: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_chart_metric(symbols: str, days: int, metric: str, session: AsyncSession) -> dict[str, Any]:
    """Get chart data for a specific metric."""
    if not symbols:
        raise HTTPException(status_code=400, detail="No symbols provided")

    # Parse symbols from comma-separated string
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

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
        try:
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

        except Exception as e:
            logger.error(f"Error fetching data for chart: {e}")
            return {"error": f"Failed to fetch {metric} data"}

    return {
        "symbols": symbol_list,
        "data": chart_data,
        "days": days,
        "metric": metric,
    }


@app.get("/api/market/top-movers")
async def get_top_movers(
    period: str = "1d", limit: int = 10, session: AsyncSession = Depends(get_db_session)
) -> dict[str, Any]:
    """Get top movers (gainers and losers) for a given period."""
    try:
        # Validate period parameter
        valid_periods = {"1d": 1, "1w": 7, "1m": 30, "90d": 90}
        if period not in valid_periods:
            raise HTTPException(status_code=400, detail="Invalid period. Use: 1d, 1w, 1m, 90d")

        days = valid_periods[period]

        # Get the latest available trading day
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
                        symbol_data[-1].quantity
                        * symbol_data[-1].current_price
                        * currency_rates[symbol_data[-1].currency]
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
                        }
                    )

        # Sort by percentage change
        movers.sort(key=lambda x: x["change_pct"], reverse=True)

        # Get top gainers and losers
        gainers = movers[:limit]
        losers = movers[-limit:][::-1]  # Reverse to show biggest losers first

        return {
            "period": period,
            "gainers": gainers,
            "losers": losers,
            "total_symbols": len(movers),
            "latest_trading_date": today.isoformat(),
            "start_date": start_date.isoformat(),
        }

    except Exception as e:
        logger.error(f"Error fetching top movers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pies")
async def get_pies(session: AsyncSession = Depends(get_db_session)):
    """Get all pies with their instruments."""
    try:
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

    except Exception as e:
        logger.error(f"Error fetching pies: {e}")
        raise HTTPException(status_code=500, detail=str(e))
