"""
Shared view-layer helpers used across multiple routers:
currency rate lookup, historical trend calculation, and the PRICE_COLUMN constant.
"""

from datetime import datetime
from typing import Optional

import numpy as np
from dateutil.relativedelta import relativedelta
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.utils.pe_history import avg_pe, basis_matches, pe_series
from config import CURRENCIES, PRICE_FIELD, TIMEZONE
from models import CurrencyRateDaily, HoldingDaily, PricesDaily

# SQLAlchemy column expression for the configured price field (used in chart and portfolio queries)
PRICE_COLUMN = getattr(PricesDaily, PRICE_FIELD.lower().replace(" ", "_") + "_price").label("price")

# Yahoo quotes GBp/GBX listings in pence but reports their marketCap in pounds.
_PENCE_QUOTE_CURRENCIES = frozenset({"GBp", "GBX"})


def statement_to_quote_factor(info: dict, rates: dict[str, float]) -> Optional[float]:
    """Multiplier converting `financialCurrency` amounts into marketCap's currency.

    Yahoo reports marketCap in the quote currency but freeCashflow, totalRevenue
    and ebitda in the reporting currency; dividing them without this is wrong for
    ~22% of the universe. None when either rate is unavailable.
    """
    quote = info.get("currency")
    statement = info.get("financialCurrency")
    if not quote or not statement:
        return None

    if quote in _PENCE_QUOTE_CURRENCIES:
        quote = "GBP"
    if quote == statement:
        return 1.0

    statement_rate = rates.get(statement)
    quote_rate = rates.get(quote)
    if not statement_rate or not quote_rate:
        return None
    return statement_rate / quote_rate


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
        "recommendation_delta_12m": None,
        "pe_1y_trend_pct": None,
        "pe_5y_avg_vs_current_pct": None,
    }

    if holding.instrument.yahoo is None:
        return trends

    # --- 1. Recommendation Trend ---
    recs = holding.instrument.yahoo.recommendations
    trends["recommendation_trend"] = 0.0
    trends["recommendation_delta_12m"] = 0.0
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
            score = (2 * sb + b - s - 2 * ss)[mask] / (2 * tot[mask])
            x = np.arange(score.size, dtype=float)
            if score.std() != 0:
                trends["recommendation_trend"] = float(np.corrcoef(x, score)[0, 1])
            # The correlation only says the drift is monotone; this says whether
            # it is large enough to be worth acting on.
            trends["recommendation_delta_12m"] = float(score[-1] - score[0])

    # --- 2. PE Trend and PE vs History ---
    pes = holding.instrument.yahoo.pes
    current_pe = holding.instrument.yahoo.info.get("trailingPE")

    # Yahoo's trailingPE and the scraped series must be on a comparable EPS basis
    # before either can be read as a re-rating.
    today = datetime.now(TIMEZONE).date()
    if pes and current_pe and current_pe > 0 and basis_matches(pes, today, current_pe):
        one_year_ago = today - relativedelta(years=1)
        past_pe = next((pe for d, pe in reversed(pe_series(pes, today, years=5)) if d <= one_year_ago), None)
        if past_pe:
            trends["pe_1y_trend_pct"] = (current_pe / past_pe - 1) * 100

        avg_pe_5y = avg_pe(pes, today)
        if avg_pe_5y:
            trends["pe_5y_avg_vs_current_pct"] = (avg_pe_5y / current_pe - 1) * 100

    return trends
