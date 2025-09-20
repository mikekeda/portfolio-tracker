from datetime import date, datetime
from typing import Dict, List, Sequence, Any

from config import TIMEZONE, PRICE_FIELD
from models import Instrument, PricesDaily


async def get_pe_history(instrument: Instrument, prices: Sequence[PricesDaily]):
    symbol = instrument.yahoo_symbol
    # --- 2. TRAILING TWELVE MONTHS (TTM) EPS CALCULATION ---

    # First, extract and sort all historical earnings reports
    sorted_earnings: List[Dict[str, Any]] = []
    current_date = datetime.now(TIMEZONE).date()
    for date_str, details in instrument.yahoo.earnings.items():
        try:
            earnings_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            reported_eps = details.get("Reported EPS")
            # Only use actual, historical earnings reports with valid EPS
            if earnings_date <= current_date and reported_eps is not None:
                sorted_earnings.append({"date": earnings_date, "eps": reported_eps})
        except (ValueError, TypeError):
            continue

    sorted_earnings.sort(key=lambda x: x["date"])

    # Now, calculate a rolling TTM EPS for each report date
    ttm_eps_by_report_date: Dict[date, float] = {}
    if len(sorted_earnings) >= 4:
        for i in range(3, len(sorted_earnings)):
            # Sum the current quarter and the previous three to get TTM EPS
            ttm_eps = sum(item["eps"] for item in sorted_earnings[i - 3 : i + 1])
            report_date = sorted_earnings[i]["date"]
            ttm_eps_by_report_date[report_date] = ttm_eps

    if not ttm_eps_by_report_date:
        return {}

    # --- 3. OPTIMIZED P/E CALCULATION LOOP ---
    pe_history = {}

    # Get a sorted list of the report dates for our efficient loop
    eps_report_dates = sorted(ttm_eps_by_report_date.keys())
    current_eps_idx = 0
    latest_ttm_eps = None

    # Single pass through price data to calculate P/E
    for price_record in prices:
        price_date = price_record.date  # type: ignore[attr-defined]

        # Efficiently find the most recent TTM EPS for the current price date
        while current_eps_idx < len(eps_report_dates) and eps_report_dates[current_eps_idx] <= price_date:
            latest_eps_date = eps_report_dates[current_eps_idx]
            latest_ttm_eps = ttm_eps_by_report_date[latest_eps_date]
            current_eps_idx += 1

        price = getattr(price_record, PRICE_FIELD.lower().replace(" ", "_") + "_price")

        # Calculate P/E if we have a valid price and a positive TTM EPS
        if price and price > 0 and latest_ttm_eps and latest_ttm_eps > 0:
            pe_ratio = price / latest_ttm_eps
            pe_history[price_date.isoformat()] = round(pe_ratio, 2)

    return pe_history
