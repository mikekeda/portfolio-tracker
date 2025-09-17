"""
FastAPI backend for Trading212 Portfolio Manager
===============================================
Serves portfolio data from PostgreSQL database.
"""

import logging
import os

# Standard library imports
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, List

# Third-party imports
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession
from sqlalchemy.orm import selectinload

from backend.screener_config import get_screener_config
from backend.utils.screener import calculate_screener_results
from backend.utils.technical import calculate_technical_indicators_for_symbols

# Local imports
from config import CURRENCIES, PRICE_FIELD, TIMEZONE, BENCH
from models import CurrencyRateDaily, HoldingDaily, Instrument, PortfolioDaily, PricesDaily

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Trading212 Portfolio API", description="API for accessing portfolio data", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _get_session_factory() -> async_sessionmaker:
    """Create and cache the async database session factory."""
    db_url = "postgresql+asyncpg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}".format(
        db_name=os.getenv("DB_NAME", "trading212_portfolio"),
        db_password=os.getenv("DB_PASSWORD"),
        db_user=os.getenv("DB_USER", "postgres"),
        db_host=os.getenv("DB_HOST", "localhost"),
        db_port=os.getenv("DB_PORT", "5432"),
    )
    engine = create_async_engine(db_url, echo=False, pool_pre_ping=True)

    return async_sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for database sessions."""
    AsyncSessionLocal = _get_session_factory()
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions."""
    async with get_session() as session:
        yield session


async def get_rates(session: AsyncSession) -> Dict[str, float]:
    """Get current currency exchange rates to GBP."""
    table = {"GBX": 0.01, "GBP": 1.0}

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


@app.get("/")
async def root() -> Dict[str, str]:
    """Health check endpoint."""
    return {"message": "Trading212 Portfolio API", "status": "running"}


@app.get("/api/portfolio/current")
async def get_current_portfolio(session: AsyncSession = Depends(get_db_session)) -> Dict[str, Any]:
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
            .options(selectinload(HoldingDaily.instrument))
        )
        holdings = holdings_result.scalars().all()

        # Get currency rates
        currency_rates = await get_rates(session)

        # Get Yahoo Finance data from database cache (optimized batch fetch)
        yahoo_profiles = {}
        yahoo_symbols = [
            holding.instrument.yahoo_symbol
            for holding in holdings
            if holding.instrument.yahoo_symbol and holding.instrument.yahoo_data
        ]

        # Fetch all Yahoo Finance data at once from the database
        instruments_result = await session.execute(
            select(Instrument).filter(Instrument.yahoo_symbol.in_(yahoo_symbols), Instrument.yahoo_data.isnot(None))
        )
        instruments = instruments_result.scalars().all()

        for instrument in instruments:
            yahoo_profiles[instrument.yahoo_symbol] = instrument.yahoo_data or {}

        # Calculate technical indicators using centralized function
        symbols_for_technical = [h.instrument.yahoo_symbol for h in holdings if h.instrument.yahoo_symbol]
        rsi_data, technical_data = await calculate_technical_indicators_for_symbols(symbols_for_technical, session)

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

            # Get Yahoo Finance data for this holding (same as terminal)
            info = yahoo_profiles[holding.instrument.yahoo_symbol]

            portfolio_data.append(
                {
                    "t212_code": holding.instrument.t212_code,
                    "name": holding.instrument.name,
                    "yahoo_symbol": holding.instrument.yahoo_symbol,
                    "currency": holding.instrument.currency,
                    "sector": holding.instrument.sector,
                    "country": holding.instrument.country,
                    "quantity": holding.quantity,
                    "avg_price": holding.avg_price,
                    "current_price": holding.current_price,
                    "ppl": holding.ppl,
                    "fx_ppl": holding.fx_ppl,
                    "market_cap": holding.market_cap,
                    "pe_ratio": holding.pe_ratio,
                    "beta": holding.beta,
                    "date": holding.date.isoformat(),
                    "market_value": market_value_gbp,  # Now in GBP
                    "profit": holding.ppl,  # Total profit (same as terminal - ppl already includes FX)
                    "return_pct": round((holding.ppl / (market_value_gbp - holding.ppl) * 100.0), 2)
                    if (market_value_gbp - holding.ppl) > 0
                    else 0.0,  # Same formula as terminal
                    "portfolio_pct": portfolio_pct,
                    # Additional fields from Yahoo Finance (same as terminal)
                    "dividend_yield": info.get("dividendYield"),
                    "business_summary": info.get("longBusinessSummary"),
                    "prediction": round((info["targetMedianPrice"] / holding.current_price - 1) * 100.0)
                    if info.get("targetMedianPrice")
                    else None,
                    "institutional_ownership": round(info["heldPercentInstitutions"] * 100.0)
                    if info.get("heldPercentInstitutions")
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
                    "free_cashflow_yield": info["freeCashflow"] / info["marketCap"] * 100
                    if (info.get("freeCashflow") and info.get("marketCap", 0) > 0)
                    else None,  # FCF / Market Cap (owner's yield)
                    "recommendation_mean": round(info["recommendationMean"], 2)
                    if info.get("recommendationMean")
                    else None,
                    "recommendation_key": info.get("recommendationKey"),
                    "number_of_analyst_opinions": info.get("numberOfAnalystOpinions"),
                    "fifty_two_week_high_distance": round(info["fiftyTwoWeekHighChangePercent"] * 100)
                    if info.get("fiftyTwoWeekHighChangePercent")
                    else None,  # Distance from 52-week high (negative = below high)
                    "fifty_two_week_change": round(info.get("52WeekChange", 0) * 100)
                    if info.get("52WeekChange") is not None
                    else None,  # True YoY change vs 52 weeks ago (positive = up from 52w ago)
                    "short_percent_of_float": info["shortPercentOfFloat"] * 100
                    if info.get("shortPercentOfFloat")
                    else None,  # Keep full precision for screener evaluation
                    "rsi": round(rsi_data[holding.instrument.yahoo_symbol]),  # RSI calculated from price history
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
            "last_updated": latest_date.isoformat(),
        }
    except Exception as e:
        logger.error(f"Error fetching current portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/summary")
async def get_portfolio_summary(session: AsyncSession = Depends(get_db_session)) -> Dict[str, Any]:
    """Get portfolio summary statistics including total value, profit, and win rate."""
    try:
        # Get the latest portfolio snapshot
        result = await session.execute(select(PortfolioDaily).order_by(PortfolioDaily.date.desc()))
        latest_snapshot = result.scalars().first()

        if not latest_snapshot:
            return {"error": "No portfolio data available"}

        # Get holdings for the same date to calculate win rate
        holdings_result = await session.execute(select(HoldingDaily).filter(HoldingDaily.date == latest_snapshot.date))
        holdings = holdings_result.scalars().all()

        profitable_count = 0
        losing_count = 0

        for holding in holdings:
            profit = holding.ppl  # Total profit (same as terminal - ppl already includes FX)
            if profit > 0:
                profitable_count += 1
            else:
                losing_count += 1

        win_rate = (profitable_count / len(holdings) * 100) if holdings else 0

        return {
            "total_value": round(latest_snapshot.total_value_gbp, 2),
            "total_profit": round(latest_snapshot.total_profit_gbp, 2),
            "total_return_pct": round(latest_snapshot.total_return_pct, 2),
            "total_holdings": len(holdings),
            "profitable_holdings": profitable_count,
            "losing_holdings": losing_count,
            "win_rate": round(win_rate, 2),
            "last_updated": latest_snapshot.date.isoformat(),
        }
    except Exception as e:
        logger.error(f"Error fetching portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/allocations")
async def get_portfolio_allocations(session: AsyncSession = Depends(get_db_session)) -> Dict[str, Any]:
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
            "total_value": latest_snapshot.total_value_gbp,
        }
    except Exception as e:
        logger.error(f"Error fetching allocations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/history")
async def get_portfolio_history(days: int = 30, session: AsyncSession = Depends(get_db_session)) -> Dict[str, Any]:
    """Get portfolio history for the last N days."""
    try:
        cutoff_date = datetime.now(TIMEZONE).date() - timedelta(days=days)
        snap_res = await session.execute(
            select(PortfolioDaily).filter(PortfolioDaily.date >= cutoff_date).order_by(PortfolioDaily.date)
        )
        snapshots = snap_res.scalars().all()

        bench_res = await session.execute(
            select(PricesDaily.date, PricesDaily.adj_close_price)
            .where(
                PricesDaily.symbol == BENCH,
                PricesDaily.date >= snapshots[0].date,  # latest date in the snapshots
            )
            .order_by(PricesDaily.date)
        )
        bench_rows = bench_res.all()
        daily_bench = {bench.date: bench.adj_close_price for bench in bench_rows}
        bench_base_price = bench_rows[0].adj_close_price
        bench_start = snapshots[0].total_return_pct
        bench = bench_start + (bench_rows[0].adj_close_price / bench_base_price - 1) * 100

        history_data = []
        for snapshot in snapshots:
            bench = (
                bench_start + (daily_bench[snapshot.date] / bench_base_price - 1) * 100
                if daily_bench.get(snapshot.date)
                else bench
            )
            history_data.append(
                {
                    "date": snapshot.date.isoformat(),
                    "total_value": snapshot.total_value_gbp,
                    "total_profit": snapshot.total_profit_gbp,
                    "total_return_pct": snapshot.total_return_pct,
                    "country_allocation": snapshot.country_allocation,
                    "sector_allocation": snapshot.sector_allocation,
                    "currency_allocation": snapshot.currency_allocation,
                    "etf_equity_split": snapshot.etf_equity_split,
                    "benchmark": BENCH,
                    "benchmark_return_pct": bench,
                }
            )

        return {"history": history_data, "days": days}
    except Exception as e:
        logger.error(f"Error fetching portfolio history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/instruments")
async def get_instruments(session: AsyncSession = Depends(get_db_session)) -> Dict[str, List[Dict[str, Any]]]:
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


@app.get("/api/chart/prices")
async def get_chart_prices(
    symbols: str, days: int = 30, session: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """Get daily price data for charting."""
    return await get_chart_metric(symbols, days, "price", session)


@app.get("/api/chart/metrics")
async def get_chart_metrics(
    symbols: str, days: int = 30, metric: str = "price", session: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """Get chart data for different metrics."""
    return await get_chart_metric(symbols, days, metric, session)


@app.get("/api/screeners")
async def get_available_screeners() -> Dict[str, Any]:
    """Get list of all available screeners with their configurations."""
    try:
        screener_config = get_screener_config()
        return screener_config.to_dict()
    except Exception as e:
        logger.error(f"Error getting available screeners: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_chart_metric(symbols: str, days: int, metric: str, session: AsyncSession) -> Dict[str, Any]:
    """Get chart data for a specific metric."""
    # Parse symbols from comma-separated string
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    if not symbols:
        raise HTTPException(status_code=400, detail="No symbols provided")

    # Calculate date range
    start_date = datetime.now(TIMEZONE).date() - timedelta(days=days)

    if metric == "price":
        # Get price data from database
        result = await session.execute(
            select(
                PricesDaily.symbol,
                PricesDaily.date,
                getattr(PricesDaily, PRICE_FIELD.lower().replace(" ", "_") + "_price").label("price"),
            ).filter(PricesDaily.symbol.in_(symbol_list), PricesDaily.date >= start_date)
        )
        price_data = result.all()

        # Convert to chart format
        chart_data: Dict[str, List[Dict[str, str | float]]] = defaultdict(list)
        for row in price_data:
            chart_data[row.symbol].append({"date": row.date.isoformat(), "value": row.price})
    else:
        # Get holdings data for other metrics
        try:
            # Get holdings data for the date range
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

            if not holdings_data:
                return {"error": "No holdings data available"}

            # Group by date and symbol
            chart_data = defaultdict(list)

            for holding in holdings_data:
                symbol = holding.instrument.yahoo_symbol
                if symbol in symbol_list:
                    if metric == "pe_ratio":
                        value = holding.pe_ratio
                    elif metric == "institutional":
                        value = holding.institutional * 100 if holding.institutional is not None else None
                    elif metric == "profit":
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

                # Sort data by date
                chart_data[symbol].sort(key=lambda x: x["date"])

        except Exception as e:
            logger.error(f"Error fetching holdings data for chart: {e}")
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
) -> Dict[str, Any]:
    """Get top movers (gainers and losers) for a given period."""
    try:
        # Validate period parameter
        valid_periods = {"1d": 1, "1w": 7, "1m": 30, "90d": 90}
        if period not in valid_periods:
            raise HTTPException(status_code=400, detail="Invalid period. Use: 1d, 1w, 1m, 90d")

        days = valid_periods[period]

        # Get the latest available trading day
        latest_date_result = await session.execute(select(func.max(PricesDaily.date)))
        latest_trading_date = latest_date_result.scalar_one()
        start_date = latest_trading_date - timedelta(days=days)

        result = await session.execute(
            select(
                Instrument.yahoo_symbol,
                Instrument.name,
                Instrument.t212_code,
                PricesDaily.symbol,
                PricesDaily.date,
                getattr(PricesDaily, PRICE_FIELD.lower().replace(" ", "_") + "_price").label("px"),
            )
            .join(PricesDaily, PricesDaily.symbol == Instrument.yahoo_symbol)
            .filter(PricesDaily.date >= start_date)
            .order_by(Instrument.yahoo_symbol, PricesDaily.date)
        )
        instruments = result.all()

        prices_data = defaultdict(list)

        for price in instruments:
            prices_data[price.symbol].append(price)

        # Calculate percentage changes
        movers = []
        for symbol, symbol_data in prices_data.items():
            if len(symbol_data) >= 2:
                # Get first and last prices
                first_price = symbol_data[0].px
                last_price = symbol_data[-1].px

                if first_price > 0:
                    change_pct = ((last_price - first_price) / first_price) * 100
                    movers.append(
                        {
                            "symbol": symbol,
                            "name": symbol_data[0].name,
                            "change_pct": round(change_pct, 2),
                            "current_price": round(last_price, 2),
                            "t212_code": symbol_data[0].t212_code,
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
            "latest_trading_date": latest_trading_date.isoformat(),
            "start_date": start_date.isoformat(),
        }

    except Exception as e:
        logger.error(f"Error fetching top movers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
