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

from config import CURRENCIES, PRICE_FIELD, TIMEZONE
from models import CurrencyRateDaily, HoldingDaily, PricesDaily

# SQLAlchemy column expression for the configured price field (used in chart and portfolio queries)
PRICE_COLUMN = getattr(PricesDaily, PRICE_FIELD.lower().replace(" ", "_") + "_price").label("price")


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
