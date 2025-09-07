"""
FastAPI backend for Trading212 Portfolio Manager
===============================================
Serves portfolio data from PostgreSQL database.
"""

# Standard library imports
import logging
from datetime import datetime, timedelta

# Third-party imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func

# Local imports
from config import PRICE_FIELD
from currency import get_currency_service
from data import ETF_COUNTRY_ALLOCATION, ETF_SECTOR_ALLOCATION
from database import get_db_service
from models import Holding, Instrument, PortfolioSnapshot
from utils.screener import calculate_screener_results
from utils.technical import calculate_technical_indicators_for_symbols
from utils.portfolio import weighted_add

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Trading212 Portfolio API",
    description="API for accessing portfolio data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get database service
db_service = get_db_service()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Trading212 Portfolio API", "status": "running"}


@app.get("/api/portfolio/current")
async def get_current_portfolio():
    """Get current portfolio holdings."""
    try:
        with db_service.get_session() as session:
            # Get the latest snapshot date
            latest_date = session.query(func.max(Holding.date)).scalar()

            if not latest_date:
                return {
                    "holdings": [],
                    "total_holdings": 0,
                    "last_updated": None
                }

            # Query holdings with instrument data in the same session
            holdings = session.query(Holding).join(Instrument).filter(
                Holding.date == latest_date
            ).order_by(Instrument.name).all()

            # Get currency rates using the centralized service
            currency_service = get_currency_service()
            currency_rates = currency_service.get_currency_table()

            # Get Yahoo Finance data from database cache (optimized batch fetch)
            yahoo_profiles = {}
            yahoo_symbols = [holding.instrument.yahoo_symbol for holding in holdings
                           if holding.instrument.yahoo_symbol and holding.instrument.yahoo_data]

            # Fetch all Yahoo Finance data at once from the database
            with db_service.get_session() as session:
                instruments = session.query(Instrument).filter(
                    Instrument.yahoo_symbol.in_(yahoo_symbols),
                    Instrument.yahoo_data.isnot(None)
                ).all()

                for instrument in instruments:
                    yahoo_profiles[instrument.yahoo_symbol] = instrument.yahoo_data

            # Calculate technical indicators using centralized function
            symbols_for_technical = [h.instrument.yahoo_symbol for h in holdings if h.instrument.yahoo_symbol]
            rsi_data, technical_data = calculate_technical_indicators_for_symbols(symbols_for_technical, db_service)

            # Calculate total portfolio value for percentage calculation
            total_portfolio_value = 0
            for holding in holdings:
                market_value_native = holding.quantity * holding.current_price
                market_value_gbp = market_value_native * currency_rates.get(holding.instrument.currency, 1.0)
                total_portfolio_value += market_value_gbp

            portfolio_data = []
            for holding in holdings:
                market_value_native = holding.quantity * holding.current_price
                market_value_gbp = market_value_native * currency_rates.get(holding.instrument.currency, 1.0)
                portfolio_pct = (market_value_gbp / total_portfolio_value * 100) if total_portfolio_value > 0 else 0

                # Get Yahoo Finance data for this holding (same as terminal)
                info = yahoo_profiles.get(holding.instrument.yahoo_symbol, {}) if holding.instrument.yahoo_symbol else {}

                portfolio_data.append({
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
                    "return_pct": round((holding.ppl / (market_value_gbp - holding.ppl) * 100.0), 2) if (market_value_gbp - holding.ppl) > 0 else 0.0,  # Same formula as terminal
                    "portfolio_pct": portfolio_pct,
                    # Additional fields from Yahoo Finance (same as terminal)
                    "prediction": round((info["targetMedianPrice"] / holding.current_price - 1) * 100.0) if info.get("targetMedianPrice") else None,
                    "institutional_ownership": round(info["heldPercentInstitutions"] * 100.0) if info.get("heldPercentInstitutions") else None,
                    "peg_ratio": info["trailingPegRatio"] if info.get("trailingPegRatio") else None,  # Keep full precision for screener evaluation
                    "profit_margins": info["profitMargins"] * 100.0 if info.get("profitMargins") else None,  # Keep full precision for screener evaluation
                    "revenue_growth": info["revenueGrowth"] * 100.0 if info.get("revenueGrowth") else None,  # Keep full precision for screener evaluation
                    "return_on_assets": info["returnOnAssets"] * 100.0 if info.get("returnOnAssets") else None,  # Keep full precision for screener evaluation
                    "return_on_equity": info["returnOnEquity"] * 100.0 if info.get("returnOnEquity") else None,  # Keep full precision for screener evaluation
                    "free_cashflow_yield": info["freeCashflow"] / info["marketCap"] * 100 if (info.get("freeCashflow") and info.get("marketCap") and info.get("marketCap") > 0) else None,  # FCF / Market Cap (owner's yield)
                    "recommendation_mean": round(info["recommendationMean"], 2) if info.get("recommendationMean") else None,
                    "fifty_two_week_high_distance": round(info["fiftyTwoWeekHighChangePercent"] * 100) if info.get("fiftyTwoWeekHighChangePercent") else None,  # Distance from 52-week high (negative = below high)
                    "fifty_two_week_change": round(info.get("52WeekChange", 0) * 100) if info.get("52WeekChange") is not None else None,  # True YoY change vs 52 weeks ago (positive = up from 52w ago)
                    "short_percent_of_float": info["shortPercentOfFloat"] * 100 if info.get("shortPercentOfFloat") else None,  # Keep full precision for screener evaluation
                    "rsi": rsi_data.get(holding.instrument.yahoo_symbol),  # RSI calculated from price history
                    "rule_of_40_score": (info.get("revenueGrowth", 0) * 100) + (info.get("profitMargins", 0) * 100) if (info.get("revenueGrowth") is not None and info.get("profitMargins") is not None) else None,  # Keep full precision
                    # Technical indicators calculated from price history
                    "sma_20": technical_data.get(holding.instrument.yahoo_symbol, {}).get('sma_20'),
                    "sma_50": technical_data.get(holding.instrument.yahoo_symbol, {}).get('sma_50'),
                    "sma_200": technical_data.get(holding.instrument.yahoo_symbol, {}).get('sma_200'),
                    "rs_6m_vs_spy": technical_data.get(holding.instrument.yahoo_symbol, {}).get('rs_6m_vs_spy'),
                    "gc_days_since": technical_data.get(holding.instrument.yahoo_symbol, {}).get('gc_days_since'),
                    "gc_within_sma50_frac": technical_data.get(holding.instrument.yahoo_symbol, {}).get('gc_within_sma50_frac'),
                    "bb_width_20": technical_data.get(holding.instrument.yahoo_symbol, {}).get('bb_width_20'),
                    "bb_width_20_p30_6m": technical_data.get(holding.instrument.yahoo_symbol, {}).get('bb_width_20_p30_6m'),
                    "vol20_lt_vol60": technical_data.get(holding.instrument.yahoo_symbol, {}).get('vol20_lt_vol60'),
                    "volume_ratio": technical_data.get(holding.instrument.yahoo_symbol, {}).get('volume_ratio'),
                    "quote_type": info.get("quoteType", "Unknown"),
                    "passedScreeners": [],  # will be populated below
                    "screener_score": 0,  # will be populated below
                })

            # Calculate screener results for each holding
            calculate_screener_results(portfolio_data)

            return {
                "holdings": portfolio_data,
                "total_holdings": len(portfolio_data),
                "last_updated": latest_date.isoformat()
            }
    except Exception as e:
        logger.error(f"Error fetching current portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/summary")
async def get_portfolio_summary():
    """Get portfolio summary statistics."""
    try:
        with db_service.get_session() as session:
            # Get the latest portfolio snapshot
            latest_snapshot = session.query(PortfolioSnapshot).order_by(
                PortfolioSnapshot.date.desc()
            ).first()

            if not latest_snapshot:
                return {"error": "No portfolio data available"}

            # Get holdings for the same date to calculate win rate
            holdings = session.query(Holding).filter(
                Holding.date == latest_snapshot.date
            ).all()

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
                "last_updated": latest_snapshot.date.isoformat()
            }
    except Exception as e:
        logger.error(f"Error fetching portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/allocations")
async def get_portfolio_allocations():
    """Get portfolio allocations by sector and country."""
    try:
        with db_service.get_session() as session:
            # Get the latest portfolio snapshot
            latest_snapshot = session.query(PortfolioSnapshot).order_by(
                PortfolioSnapshot.date.desc()
            ).first()

            if not latest_snapshot:
                return {"error": "No portfolio data available"}

            # Use the stored allocations from the snapshot if available
            if latest_snapshot.sector_allocation and latest_snapshot.country_allocation:
                return {
                    "sector_allocation": latest_snapshot.sector_allocation,
                    "country_allocation": latest_snapshot.country_allocation,
                    "total_value": latest_snapshot.total_value_gbp
                }

            # Fallback: calculate from holdings (same logic as terminal)
            holdings = session.query(Holding).join(Instrument).filter(
                Holding.date == latest_snapshot.date
            ).all()

            if not holdings:
                return {"error": "No portfolio data available"}

            # Get currency rates using the centralized service
            currency_service = get_currency_service()
            currency_rates = currency_service.get_currency_table()

            sector_allocation = {}
            country_allocation = {}

            for holding in holdings:
                # Calculate GBP value (same as terminal)
                market_value_native = holding.quantity * holding.current_price
                market_value_gbp = market_value_native * currency_rates.get(holding.instrument.currency, 1.0)

                # Get symbol for ETF allocation lookup
                symbol = holding.instrument.yahoo_symbol

                # Apply ETF allocation logic (same as terminal)
                if symbol in ETF_SECTOR_ALLOCATION:
                    weighted_add(sector_allocation, ETF_SECTOR_ALLOCATION[symbol], market_value_gbp)
                else:
                    sector = holding.instrument.sector or "Other"
                    sector_allocation[sector] = sector_allocation.get(sector, 0) + market_value_gbp

                if symbol in ETF_COUNTRY_ALLOCATION:
                    weighted_add(country_allocation, ETF_COUNTRY_ALLOCATION[symbol], market_value_gbp)
                else:
                    country = holding.instrument.country or "Other"
                    country_allocation[country] = country_allocation.get(country, 0) + market_value_gbp

            # Convert to percentages (same as terminal)
            total_value = sum(sector_allocation.values())

            sector_pct = {k: round((v / total_value) * 100, 2) for k, v in sorted(sector_allocation.items(), key=lambda kv: kv[1], reverse=True)}
            country_pct = {k: round((v / total_value) * 100, 2) for k, v in sorted(country_allocation.items(), key=lambda kv: kv[1], reverse=True)}

            return {
                "sector_allocation": sector_pct,
                "country_allocation": country_pct,
                "total_value": latest_snapshot.total_value_gbp  # Use correct GBP value
            }
    except Exception as e:
        logger.error(f"Error fetching allocations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/history")
async def get_portfolio_history(days: int = 30):
    """Get portfolio history for the last N days."""
    try:
        snapshots = db_service.get_portfolio_history(days)

        history_data = []
        for snapshot in snapshots:
            history_data.append({
                "date": snapshot.date.isoformat(),
                "total_value": snapshot.total_value_gbp,
                "total_profit": snapshot.total_profit_gbp,
                "total_return_pct": snapshot.total_return_pct,
                "country_allocation": snapshot.country_allocation,
                "sector_allocation": snapshot.sector_allocation
            })

        return {
            "history": history_data,
            "days": days
        }
    except Exception as e:
        logger.error(f"Error fetching portfolio history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/instruments")
async def get_instruments():
    """Get all instruments in the database for autocomplete."""
    try:
        with db_service.get_session() as session:
            instruments = session.query(Instrument).filter(
                Instrument.yahoo_symbol.isnot(None)
            ).order_by(Instrument.name).all()

            return {
                "instruments": [
                    {
                        "id": instrument.id,
                        "symbol": instrument.yahoo_symbol,
                        "name": instrument.name,
                        "t212_code": instrument.t212_code
                    }
                    for instrument in instruments
                ]
            }
    except Exception as e:
        logger.error(f"Error fetching instruments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chart/prices")
async def get_chart_prices(symbols: str, days: int = 30):
    """Get daily price data for charting."""
    return await get_chart_metric(symbols, days, "price")


@app.get("/api/chart/metrics")
async def get_chart_metrics(symbols: str, days: int = 30, metric: str = "price"):
    """Get chart data for different metrics: price, pe_ratio, institutional, profit, profit_pct."""
    return await get_chart_metric(symbols, days, metric)


@app.get("/api/screeners")
async def get_available_screeners():
    """Get list of all available screeners with their configurations."""
    try:
        from screener_config import get_screener_config
        screener_config = get_screener_config()
        return screener_config.to_dict()
    except Exception as e:
        logger.error(f"Error getting available screeners: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_chart_metric(symbols: str, days: int, metric: str):
    """Get chart data for a specific metric."""
    # Parse symbols from comma-separated string
    symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]

    if not symbol_list:
        raise HTTPException(status_code=400, detail="No symbols provided")

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    if metric == "price":
        # Get price data from database service
        price_data = db_service.get_price_history(
            tickers=symbol_list,
            start=start_date,
            end=end_date,
            price_field=PRICE_FIELD,
        )

        if price_data.empty:
            return {"error": "No price data available"}

        # Convert to chart format
        chart_data = {}
        for symbol in symbol_list:
            if symbol in price_data.columns:
                symbol_data = price_data[symbol].dropna()
                chart_data[symbol] = [
                    {
                        "date": date.isoformat(),
                        "value": float(price)
                    }
                    for date, price in symbol_data.items()
                    if price is not None and price > 0
                ]
            else:
                chart_data[symbol] = []
    else:
        # Get holdings data for other metrics
        try:
            with db_service.get_session() as session:
                # Get holdings data for the date range
                holdings_data = session.query(Holding).join(Instrument).filter(
                    Instrument.yahoo_symbol.in_(symbol_list),
                    Holding.date >= start_date,
                    Holding.date <= end_date
                ).order_by(Holding.date).all()

                if not holdings_data:
                    return {"error": "No holdings data available"}

                # Group by date and symbol
                chart_data = {}
                for symbol in symbol_list:
                    chart_data[symbol] = []

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
                            value = round((holding.ppl / (market_value - holding.ppl) * 100.0), 2) if (market_value - holding.ppl) > 0 else 0.0
                        else:
                            value = None

                        if value is not None:
                            chart_data[symbol].append({
                                "date": holding.date.isoformat(),
                                "value": float(value)
                            })

                # Sort data by date for each symbol
                for symbol in chart_data:
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
