from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app import get_db_session
from backend.screener_config import get_screener_config
from backend.utils.roic import get_roic
from backend.views._shared import PRICE_COLUMN
from config import TIMEZONE
from models import HoldingDaily, Instrument, InstrumentMetricsDaily, PricesDaily

router = APIRouter()


@router.get("/api/chart/prices")
async def get_chart_prices(
    symbols: str, days: int = 30, session: AsyncSession = Depends(get_db_session)
) -> dict[str, Any]:
    """Get daily price data for charting."""
    return await _get_chart_metric(symbols, days, "price", session)


@router.get("/api/chart/metrics")
async def get_chart_metrics(
    symbols: str, days: int = 30, metric: str = "price", session: AsyncSession = Depends(get_db_session)
) -> dict[str, Any]:
    """Get chart data for different metrics."""
    return await _get_chart_metric(symbols, days, metric, session)


@router.get("/api/screeners")
async def get_available_screeners() -> dict[str, Any]:
    """Get list of all available screeners with their configurations."""
    screener_config = get_screener_config()
    return screener_config.to_dict()


_FUNDAMENTAL_KEYS = (
    "price",
    "currency",
    "market_cap",
    "pe",
    "fwd_pe",
    "peg",
    "roic",
    "revenue_growth",
    "profit_margins",
    "fifty_two_week_high_distance",
)


def _empty_fundamentals() -> dict[str, Any]:
    return {key: None for key in _FUNDAMENTAL_KEYS}


def _scale_pct(value: Any) -> float | None:
    return value * 100.0 if value is not None else None


@router.get("/api/chart/fundamentals")
async def get_chart_fundamentals(symbols: str, session: AsyncSession = Depends(get_db_session)) -> dict[str, Any]:
    """Return a compact set of fundamentals per symbol for the chart summary row.

    Fields: price, currency, market_cap, pe, fwd_pe, peg, roic, revenue_growth,
    profit_margins, fifty_two_week_high_distance (negative %, 0 = at the high).
    Missing values are returned as null so ETFs/benchmarks degrade gracefully.
    """
    if not symbols:
        raise HTTPException(status_code=400, detail="No symbols provided")

    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        return {"data": {}}

    result = await session.execute(
        select(Instrument).where(Instrument.yahoo_symbol.in_(symbol_list)).options(selectinload(Instrument.yahoo))
    )
    instruments = result.scalars().all()

    data: dict[str, dict[str, Any]] = {sym: _empty_fundamentals() for sym in symbol_list}
    for instrument in instruments:
        info = (instrument.yahoo.info or {}) if instrument.yahoo else {}
        data[instrument.yahoo_symbol] = {
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "currency": info.get("currency"),
            "market_cap": info.get("marketCap"),
            "pe": info.get("trailingPE"),
            "fwd_pe": info.get("forwardPE"),
            "peg": info.get("trailingPegRatio"),
            "roic": get_roic(info),
            "revenue_growth": _scale_pct(info.get("revenueGrowth")),
            "profit_margins": _scale_pct(info.get("profitMargins")),
            "fifty_two_week_high_distance": _scale_pct(info.get("fiftyTwoWeekHighChangePercent")),
        }

    return {"symbols": symbol_list, "data": data}


async def _get_chart_metric(symbols: str, days: int, metric: str, session: AsyncSession) -> dict[str, Any]:
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
