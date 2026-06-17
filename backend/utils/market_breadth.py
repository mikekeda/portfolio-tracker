"""7-day rolling mean of stored daily market breadth (weekday snapshots)."""

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import TIMEZONE
from models import MarketMetricsDaily

BREADTH_ROLLING_WINDOW = 7


def rolling_mean(values: list[float | None], window: int = BREADTH_ROLLING_WINDOW) -> list[float | None]:
    """Trailing mean over the last `window` non-null values ending at each index."""
    out: list[float | None] = []
    for i in range(len(values)):
        chunk = [v for v in values[max(0, i - window + 1): i + 1] if v is not None]
        out.append(sum(chunk) / len(chunk) if chunk else None)
    return out


def rolled_breadth_series(rows: list[Any], window: int = BREADTH_ROLLING_WINDOW) -> list[float | None]:
    """Map each metrics row to rolling mean of raw daily breadth (weekday rows only)."""
    weekday_raw = [row.market_breadth_indicator for row in rows if row.date.weekday() < 5]
    weekday_rolled = rolling_mean(weekday_raw, window)

    rolled: list[float | None] = []
    wi = 0
    last: float | None = None
    for row in rows:
        if row.date.weekday() < 5:
            last = weekday_rolled[wi]
            wi += 1
        rolled.append(last)
    return rolled


async def compute_live_rolling_breadth(
    session: AsyncSession,
    live_raw: float,
    window: int = BREADTH_ROLLING_WINDOW,
) -> float:
    """Rolling mean of stored weekday raw breadth plus today's live reading."""
    today = datetime.now(TIMEZONE).date()
    result = await session.execute(
        select(MarketMetricsDaily.date, MarketMetricsDaily.market_breadth_indicator)
        .where(
            MarketMetricsDaily.market_breadth_indicator.isnot(None),
            MarketMetricsDaily.date < today,
        )
        .order_by(MarketMetricsDaily.date.desc())
        .limit(40)
    )
    prior = [row.market_breadth_indicator for row in result if row.date.weekday() < 5]
    prior.reverse()
    chunk = [v for v in (prior + [live_raw])[-window:] if v is not None]
    return sum(chunk) / len(chunk) if chunk else live_raw
