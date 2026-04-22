from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import aiohttp
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app import get_db_session
from backend.utils.market_data import gen_fred_latest
from backend.views._shared import get_rates
from config import TIMEZONE
from models import HoldingDaily, Instrument, MarketMetricsDaily, Pie, PieInstrument

router = APIRouter()


@router.get("/api/market/movers")
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

    # Resolve a single sector/country label per holding for the Asset Allocation
    # treemap. ETFs are pro-rata split across many groups in the portfolio-wide
    # allocation snapshot, so for a per-tile view we tag them as 'Fund' for the
    # sector facet and fall back to Yahoo's country (often missing for global
    # ETFs) -> 'Global' for the region facet, rather than picking an arbitrary
    # primary group.
    metadata: dict[str, dict[str, str]] = {}
    if prices_data:
        inst_result = await session.execute(
            select(Instrument)
            .where(Instrument.yahoo_symbol.in_(list(prices_data.keys())))
            .options(selectinload(Instrument.yahoo))
        )
        for inst in inst_result.scalars().all():
            info = (inst.yahoo.info or {}) if inst.yahoo else {}
            if info.get("quoteType") == "ETF":
                metadata[inst.yahoo_symbol] = {
                    "sector": "Fund",
                    "country": info.get("country") or "Global",
                }
            else:
                metadata[inst.yahoo_symbol] = {
                    "sector": info.get("sector") or "Other",
                    "country": info.get("country") or "Other",
                }

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

                meta = metadata.get(symbol, {"sector": "Other", "country": "Other"})
                movers.append(
                    {
                        "symbol": symbol,
                        "name": symbol_data[0].name,
                        "change_pct": change_pct,
                        "current_price": last_price,
                        "t212_code": symbol_data[0].t212_code,
                        "gain_pct": gain_pct,
                        "value": market_value_gbp,
                        "sector": meta["sector"],
                        "country": meta["country"],
                    }
                )

    # Sort by percentage change (descending: highest gainers first)
    movers.sort(key=lambda x: x["change_pct"], reverse=True)

    return movers


@router.get("/api/market/indicators/history")
async def get_market_indicators_history(
    days: int = 365,
    session: AsyncSession = Depends(get_db_session),
) -> list[dict[str, Any]]:
    """
    Returns historical market metrics for charting.
    days=0 returns all available history.
    market_breadth_indicator is returned pre-multiplied by 100 (as a percent).
    consumer_sentiment is included when stored (nullable until the migration is applied and the
    script has run at least once).
    """
    if days == 0:
        query = select(MarketMetricsDaily).order_by(MarketMetricsDaily.date)
    else:
        cutoff = datetime.now(TIMEZONE).date() - timedelta(days=days)
        query = select(MarketMetricsDaily).where(MarketMetricsDaily.date >= cutoff).order_by(MarketMetricsDaily.date)

    result = await session.execute(query)
    rows = result.scalars().all()
    return [
        {
            "date": row.date.isoformat(),
            "buffett_indicator": row.buffett_indicator,
            "yield_spread": row.yield_spread,
            "fear_greed_index": row.fear_greed_index,
            "vix": row.vix,
            "market_breadth_indicator": (
                round(row.market_breadth_indicator * 100, 2) if row.market_breadth_indicator is not None else None
            ),
            "sp500_above_sma200": row.sp500_above_sma200,
            "consumer_sentiment": row.consumer_sentiment,
        }
        for row in rows
    ]


@router.get("/api/market/indicators/consumer-sentiment/history")
async def get_consumer_sentiment_history(months: int = 36) -> list[dict[str, Any]]:
    """
    Fetch historical University of Michigan Consumer Sentiment (UMCSENT) from FRED.
    Returns oldest-first so charts render chronologically.
    months=0 returns ~10 years (120 observations).
    """
    limit = 120 if months == 0 else max(3, months)
    async with aiohttp.ClientSession() as http_session:
        obs = await gen_fred_latest(http_session, "UMCSENT", limit=limit)
    if not obs:
        return []
    # gen_fred_latest returns newest-first; reverse for chart chronological order
    return [{"date": o["date"], "value": o["value"]} for o in reversed(obs)]


@router.get("/api/pies")
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
