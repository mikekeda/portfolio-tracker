"""Currency-rate lookups shared by the view layer and the DCF model."""

from datetime import datetime
from typing import Iterable, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import TIMEZONE, logger
from models import CurrencyRateDaily

# Past a long weekend the rate job has likely stopped. Valuing on the last known
# rate still beats refusing to value, but it should not pass unnoticed.
MAX_FX_STALENESS_DAYS = 3

# Anchoring the scan on the newest row rather than on today keeps DISTINCT ON off
# a full sort (19.5ms -> 0.3ms) without reintroducing a dependency on today's row.
FX_SCAN_WINDOW_DAYS = 30


async def latest_rates_to_gbp(
    session: AsyncSession, currencies: Optional[Iterable[str]] = None
) -> dict[str, float]:
    """Most recent rate into GBP per source currency; all currencies when None.

    Keyed on the latest available date rather than today's, so a late or missed
    rate job degrades to a slightly stale valuation instead of none at all.
    """
    newest = select(func.max(CurrencyRateDaily.date)).where(CurrencyRateDaily.to_currency == "GBP").scalar_subquery()
    query = (
        select(CurrencyRateDaily.from_currency, CurrencyRateDaily.rate, CurrencyRateDaily.date)
        .where(
            CurrencyRateDaily.to_currency == "GBP",
            CurrencyRateDaily.date >= newest - FX_SCAN_WINDOW_DAYS,
        )
        .distinct(CurrencyRateDaily.from_currency)
        .order_by(CurrencyRateDaily.from_currency, CurrencyRateDaily.date.desc())
    )
    if currencies is not None:
        query = query.where(CurrencyRateDaily.from_currency.in_(list(currencies)))

    rows = (await session.execute(query)).all()

    oldest = min((row.date for row in rows), default=None)
    if oldest is not None and (datetime.now(TIMEZONE).date() - oldest).days > MAX_FX_STALENESS_DAYS:
        logger.warning("Currency rates are stale — oldest is %s", oldest)

    return {row.from_currency: row.rate for row in rows}
