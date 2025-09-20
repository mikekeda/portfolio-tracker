from datetime import date, datetime, timedelta
from typing import Dict, List, Sequence, Any, Optional

from config import TIMEZONE, PRICE_FIELD
from models import Instrument, PricesDaily


def _safe_float(value: Any) -> Optional[float]:
    """Safely convert a value to a float, returning None if it's not a valid number."""
    if isinstance(value, (int, float)):
        return float(value)
    return None


async def get_pe_history(instrument: Instrument, prices: Sequence[PricesDaily]) -> Dict[str, float]:
    """
    Calculates the historical Trailing Twelve Months (TTM) P/E ratio for a given instrument.

    This function reconstructs the TTM EPS at each point in time based on historical
    earnings reports and then divides the historical stock price by the corresponding TTM EPS.

    Args:
        instrument: The Instrument object, containing cached earnings data.
        prices: A sequence of historical daily prices for the instrument.

    Returns:
        A dictionary where keys are date strings (ISO format) and values are
        the calculated P/E ratios for those dates. Returns an empty dictionary
        if P/E history cannot be calculated.
    """
    if not instrument.yahoo or not instrument.yahoo.earnings:
        return {}

    # --- 1. Extract and Clean Historical Earnings Data ---
    sorted_earnings: List[Dict[str, Any]] = []
    current_date = datetime.now(TIMEZONE).date()
    for date_str, details in instrument.yahoo.earnings.items():
        try:
            earnings_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            reported_eps = _safe_float(details.get("Reported EPS"))

            # Only use actual, historical earnings reports with a valid EPS value
            if earnings_date <= current_date and reported_eps is not None:
                sorted_earnings.append({"date": earnings_date, "eps": reported_eps})
        except (ValueError, TypeError):
            # Ignore entries where the key is not a valid date string
            continue

    if not sorted_earnings:
        return {}

    sorted_earnings.sort(key=lambda x: x["date"])

    # --- 2. Calculate Rolling TTM EPS (More Robust Method) ---
    # This method is more robust to irregular reporting schedules than a fixed 4-quarter sum.
    ttm_eps_by_report_date: Dict[date, float] = {}
    for i, current_report in enumerate(sorted_earnings):
        report_date = current_report["date"]
        # Define the one-year window for TTM calculation
        one_year_prior = report_date - timedelta(days=365)

        # Sum EPS from all reports within the last year (inclusive)
        ttm_eps = sum(
            report["eps"]
            for report in sorted_earnings[: i + 1]  # Only look at past and current reports
            if report["date"] > one_year_prior
        )

        ttm_eps_by_report_date[report_date] = ttm_eps

    if not ttm_eps_by_report_date:
        return {}

    # --- 3. Map Historical Prices to TTM EPS to Calculate P/E ---
    pe_history = {}
    eps_report_dates = sorted(ttm_eps_by_report_date.keys())
    current_eps_idx = 0
    latest_ttm_eps: Optional[float] = None

    for price_record in prices:
        price_date = price_record.date

        # Efficiently find the most recent TTM EPS value for the current price date
        while current_eps_idx < len(eps_report_dates) and eps_report_dates[current_eps_idx] <= price_date:
            latest_eps_date = eps_report_dates[current_eps_idx]
            latest_ttm_eps = ttm_eps_by_report_date[latest_eps_date]
            current_eps_idx += 1

        price = getattr(price_record, PRICE_FIELD.lower().replace(" ", "_") + "_price")

        # Calculate P/E only if we have a valid price and a positive TTM EPS
        if price and price > 0 and latest_ttm_eps and latest_ttm_eps > 0:
            pe_ratio = price / latest_ttm_eps
            pe_history[price_date.isoformat()] = round(pe_ratio, 2)

    return pe_history
