"""
Shared view-layer helpers used across multiple routers:
currency rate lookup, historical trend calculation, and the PRICE_COLUMN constant.
"""

from datetime import datetime
from typing import Optional

import numpy as np
from dateutil.relativedelta import relativedelta
from sqlalchemy.ext.asyncio import AsyncSession

from backend.utils.fx import latest_rates_to_gbp
from backend.utils.pe_history import basis_matches, harmonic_mean_pe, pe_series
from config import CURRENCIES, PRICE_FIELD, TIMEZONE
from models import HoldingDaily, PricesDaily

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
    """Latest available exchange rate into GBP per currency, including pence units.

    Callers must subscript the result rather than defaulting a miss to 1.0 —
    valuing a foreign holding at parity silently inflates the whole portfolio.
    """
    return {"GBX": 0.01, "GBP": 1.0, "GBp": 0.01, **await latest_rates_to_gbp(session, CURRENCIES)}


def calculate_historical_trends(holding: HoldingDaily) -> dict[str, Optional[float]]:
    """Recommendation and P/E trend metrics from the yahoo object's history.

    Also returns `avg_pe`, the 5-year representative multiple: it falls out of
    the same parse as the trends, and every key here is spliced into the
    holdings response, so callers need not compute it separately.
    """
    trends: dict[str, Optional[float]] = {
        "recommendation_trend": None,
        "recommendation_delta_12m": None,
        "pe_1y_trend_pct": None,
        "pe_5y_avg_vs_current_pct": None,
        "avg_pe": None,
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

    if not pes:
        return trends

    # Parsing `pes` dominated this function, so the basis check, the 1y trend and
    # the 5y average all read one series rather than reparsing per statistic.
    today = datetime.now(TIMEZONE).date()
    series = pe_series(pes, today, years=5)
    avg_pe_5y = harmonic_mean_pe([pe for _, pe in series])
    trends["avg_pe"] = avg_pe_5y

    # Yahoo's trailingPE and the scraped series must be on a comparable EPS basis
    # before either can be read as a re-rating.
    if current_pe and current_pe > 0 and basis_matches(series, today, current_pe):
        one_year_ago = today - relativedelta(years=1)
        past_pe = next((pe for d, pe in reversed(series) if d <= one_year_ago), None)
        if past_pe:
            trends["pe_1y_trend_pct"] = (current_pe / past_pe - 1) * 100

        if avg_pe_5y:
            trends["pe_5y_avg_vs_current_pct"] = (avg_pe_5y / current_pe - 1) * 100

    return trends
