"""Piotroski F-Score helper utilities.

Pure functions that compute a 9-component Piotroski F-Score from cached Yahoo
annual statements. No DB or network I/O — callers pass the JSONB blobs from
:class:`models.InstrumentYahoo` directly.

Statement payloads are shaped ``{iso_date_str: {row_label: value}}`` (i.e. the
output of ``DataFrame.to_dict()`` where columns are fiscal-year-end dates and
rows are line items). The two most recent dates are used for the YoY checks.
"""

from typing import Any, Optional, TypedDict


# Human-readable check labels — these strings end up in the Holdings tooltip
# (as `✓ {label}` / `✗ {label}`) and the Stock page breakdown, so keep them
# short and self-explanatory.
CHECK_LABELS: dict[str, str] = {
    # Profitability (4)
    "net_income_positive": "Net income > 0",
    "ocf_positive": "Operating cash flow > 0",
    "roa_increasing": "ROA increasing YoY",
    "ocf_gt_net_income": "OCF > net income (quality)",
    # Leverage / Liquidity / Source of Funds (3)
    "leverage_decreasing": "Long-term debt / assets decreasing",
    "current_ratio_increasing": "Current ratio increasing YoY",
    "no_new_shares": "No net share issuance",
    # Operating efficiency (2)
    "gross_margin_increasing": "Gross margin increasing YoY",
    "asset_turnover_increasing": "Asset turnover increasing YoY",
}


class FScoreResult(TypedDict):
    score: int
    details: dict[str, bool]
    available: int


def _safe_number(x: Any) -> Optional[float]:
    """Convert value to float, returning None for None / NaN / inf / non-numeric."""
    if not isinstance(x, (int, float)):
        return None
    if isinstance(x, float) and (x != x or x == float("inf") or x == float("-inf")):
        return None
    return float(x)


def _row(statement: Optional[dict[str, Any]], date_key: str) -> dict[str, Any]:
    """Return the {row_label: value} dict for a fiscal year, or empty if missing."""
    if not isinstance(statement, dict):
        return {}
    val = statement.get(date_key)
    return val if isinstance(val, dict) else {}


def _latest_two_keys(statement: Optional[dict[str, Any]]) -> tuple[Optional[str], Optional[str]]:
    """Return (current_year_key, prior_year_key) — sorted ISO date strings sort chronologically."""
    if not isinstance(statement, dict) or len(statement) < 2:
        return None, None
    keys = sorted(statement.keys())
    return keys[-1], keys[-2]


def _ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return num / den


def get_piotroski_f_score(
    cashflow: Optional[dict[str, Any]],
    balance_sheet: Optional[dict[str, Any]],
    income_stmt: Optional[dict[str, Any]],
) -> Optional[FScoreResult]:
    """Compute the 9-component Piotroski F-Score.

    Returns None when there isn't enough data to score at all (e.g. ETFs with
    no financial statements, or only a single fiscal year cached). Otherwise
    returns the integer score (0-9), a per-check breakdown, and `available`
    — the count of checks for which complete data was found. Checks that
    couldn't be computed count as failures in `score` (so a stock with sparse
    data won't artificially top the leaderboard); `available` lets callers
    flag a low-confidence reading.
    """
    cur, prev = _latest_two_keys(income_stmt)
    bs_cur, bs_prev = _latest_two_keys(balance_sheet)
    cf_cur, _ = _latest_two_keys(cashflow)

    # Need at least two fiscal years of income + balance sheet for any YoY check.
    if cur is None or prev is None or bs_cur is None or bs_prev is None:
        return None

    inc_cur = _row(income_stmt, cur)
    inc_prev = _row(income_stmt, prev)
    bs_cur_row = _row(balance_sheet, bs_cur)
    bs_prev_row = _row(balance_sheet, bs_prev)
    cf_cur_row = _row(cashflow, cf_cur) if cf_cur else {}

    net_income_cur = _safe_number(inc_cur.get("Net Income"))
    net_income_prev = _safe_number(inc_prev.get("Net Income"))
    revenue_cur = _safe_number(inc_cur.get("Total Revenue"))
    revenue_prev = _safe_number(inc_prev.get("Total Revenue"))
    gross_profit_cur = _safe_number(inc_cur.get("Gross Profit"))
    gross_profit_prev = _safe_number(inc_prev.get("Gross Profit"))

    total_assets_cur = _safe_number(bs_cur_row.get("Total Assets"))
    total_assets_prev = _safe_number(bs_prev_row.get("Total Assets"))
    current_assets_cur = _safe_number(bs_cur_row.get("Current Assets"))
    current_assets_prev = _safe_number(bs_prev_row.get("Current Assets"))
    current_liab_cur = _safe_number(bs_cur_row.get("Current Liabilities"))
    current_liab_prev = _safe_number(bs_prev_row.get("Current Liabilities"))
    ltd_cur = _safe_number(bs_cur_row.get("Long Term Debt"))
    ltd_prev = _safe_number(bs_prev_row.get("Long Term Debt"))
    shares_cur = _safe_number(bs_cur_row.get("Ordinary Shares Number") or bs_cur_row.get("Share Issued"))
    shares_prev = _safe_number(bs_prev_row.get("Ordinary Shares Number") or bs_prev_row.get("Share Issued"))

    ocf_cur = _safe_number(cf_cur_row.get("Operating Cash Flow"))

    details: dict[str, bool] = {}
    available = 0

    def _record(check_id: str, passed: Optional[bool]) -> None:
        """Treat insufficient data as a failed check, but exclude it from `available`."""
        nonlocal available
        if passed is None:
            details[CHECK_LABELS[check_id]] = False
        else:
            details[CHECK_LABELS[check_id]] = passed
            available += 1

    # --- Profitability ---
    _record("net_income_positive", (net_income_cur > 0) if net_income_cur is not None else None)
    _record("ocf_positive", (ocf_cur > 0) if ocf_cur is not None else None)

    roa_cur = _ratio(net_income_cur, total_assets_cur)
    roa_prev = _ratio(net_income_prev, total_assets_prev)
    _record(
        "roa_increasing",
        (roa_cur > roa_prev) if (roa_cur is not None and roa_prev is not None) else None,
    )

    _record(
        "ocf_gt_net_income",
        (ocf_cur > net_income_cur) if (ocf_cur is not None and net_income_cur is not None) else None,
    )

    # --- Leverage / Liquidity / Source of Funds ---
    # Original Piotroski uses the LTD/Total Assets *ratio*, not absolute LTD —
    # otherwise companies that grow assets via debt look the same as ones
    # that pay down debt.
    lev_cur = _ratio(ltd_cur, total_assets_cur)
    lev_prev = _ratio(ltd_prev, total_assets_prev)
    _record(
        "leverage_decreasing",
        (lev_cur < lev_prev) if (lev_cur is not None and lev_prev is not None) else None,
    )

    cr_cur = _ratio(current_assets_cur, current_liab_cur)
    cr_prev = _ratio(current_assets_prev, current_liab_prev)
    _record(
        "current_ratio_increasing",
        (cr_cur > cr_prev) if (cr_cur is not None and cr_prev is not None) else None,
    )

    # Buybacks count in the company's favour: passes if shares didn't grow.
    _record(
        "no_new_shares",
        (shares_cur <= shares_prev) if (shares_cur is not None and shares_prev is not None) else None,
    )

    # --- Operating efficiency ---
    gm_cur = _ratio(gross_profit_cur, revenue_cur)
    gm_prev = _ratio(gross_profit_prev, revenue_prev)
    _record(
        "gross_margin_increasing",
        (gm_cur > gm_prev) if (gm_cur is not None and gm_prev is not None) else None,
    )

    at_cur = _ratio(revenue_cur, total_assets_cur)
    at_prev = _ratio(revenue_prev, total_assets_prev)
    _record(
        "asset_turnover_increasing",
        (at_cur > at_prev) if (at_cur is not None and at_prev is not None) else None,
    )

    # If literally none of the nine checks had usable data, the cached
    # statements are too sparse to score honestly.
    if available == 0:
        return None

    return {
        "score": sum(1 for v in details.values() if v),
        "details": details,
        "available": available,
    }
