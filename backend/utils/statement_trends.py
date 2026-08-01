"""Multi-period trends from quarterly statements.

Everything else in the fundamentals stack reads a single TTM snapshot, which
cannot tell a decade of compounding from one cyclical rebound. These helpers are
the only place a direction is measured.

Pure stdlib — keep it importable without the venv so `tests/` can cover it.
"""

from typing import Any, Optional

QUARTERS_PER_YEAR = 4
# Trends are measured as newest-minus-oldest over four quarters rather than an
# OLS slope: four points is too few for a regression to mean anything.
TREND_QUARTERS = 4
# Below two YoY pairs the average is just one comparison, which `revenue_growth`
# already carries.
MIN_YOY_PAIRS = 2


def sorted_periods(statement: Optional[dict[str, Any]]) -> list[str]:
    """Period keys newest first."""
    if not statement:
        return []
    return sorted(statement, reverse=True)


def _margin(is_stmt: dict[str, Any], numerator_keys: tuple[str, ...]) -> Optional[float]:
    """Margin as a percentage of revenue for one period."""
    revenue = is_stmt.get("Total Revenue") or is_stmt.get("Operating Revenue")
    if not revenue:
        return None

    for key in numerator_keys:
        value = is_stmt.get(key)
        if value is not None:
            return value / revenue * 100.0
    return None


def margin_series(
    quarterly_income_stmt: Optional[dict[str, Any]],
    numerator_keys: tuple[str, ...],
    quarters: int = TREND_QUARTERS,
) -> list[float]:
    """Margin percentages, newest first. Empty unless every quarter is computable."""
    periods = sorted_periods(quarterly_income_stmt)[:quarters]
    if len(periods) < quarters:
        return []

    series = [_margin(quarterly_income_stmt[p], numerator_keys) for p in periods]  # type: ignore[index]
    return [] if any(v is None for v in series) else series  # type: ignore[misc]


def trend(series: list[float]) -> Optional[float]:
    """Change from oldest to newest across the series, in the series' own units.

    Positive means improving. None when the series is too short to have a
    direction — never 0, which would read as "flat".
    """
    if len(series) < 2:
        return None
    return series[0] - series[-1]


def gross_margin_trend(quarterly_income_stmt: Optional[dict[str, Any]]) -> Optional[float]:
    """Percentage-point change in gross margin over the last four quarters."""
    return trend(margin_series(quarterly_income_stmt, ("Gross Profit",)))


def operating_margin_trend(quarterly_income_stmt: Optional[dict[str, Any]]) -> Optional[float]:
    """Percentage-point change in operating margin over the last four quarters."""
    return trend(margin_series(quarterly_income_stmt, ("Operating Income", "EBIT")))


def revenue_growth_yoy_avg(quarterly_income_stmt: Optional[dict[str, Any]]) -> Optional[float]:
    """Mean year-on-year quarterly revenue growth, as a percentage.

    Averages up to four YoY comparisons to smooth the single-period growth that
    makes a cyclical rebound look like durable compounding.

    Yahoo populates revenue for only 5 quarters however many columns it returns,
    which yields exactly one pair — matching `revenue_growth` to within 1pp for
    91 of 110 instruments measured. Hence MIN_YOY_PAIRS: below it this restates a
    field we already have. Stays None until merged history reaches 6 quarters.
    """
    periods = sorted_periods(quarterly_income_stmt)

    rates: list[float] = []
    for i in range(TREND_QUARTERS):
        if i + QUARTERS_PER_YEAR >= len(periods):
            break
        current = quarterly_income_stmt[periods[i]]  # type: ignore[index]
        year_ago = quarterly_income_stmt[periods[i + QUARTERS_PER_YEAR]]  # type: ignore[index]
        now = current.get("Total Revenue") or current.get("Operating Revenue")
        then = year_ago.get("Total Revenue") or year_ago.get("Operating Revenue")
        if not now or not then or then <= 0:
            break
        rates.append((now / then - 1) * 100.0)

    if len(rates) < MIN_YOY_PAIRS:
        return None
    return sum(rates) / len(rates)


def eps_next_quarter_growth(estimates: Optional[dict[str, Any]], horizon: str = "+1q") -> Optional[float]:
    """Consensus EPS growth for the next quarter, as a percentage."""
    row = ((estimates or {}).get("earnings_estimate") or {}).get(horizon)
    if not row:
        return None

    growth = row.get("growth")
    return growth * 100.0 if growth is not None else None


def eps_revision_ratio(estimates: Optional[dict[str, Any]], horizon: str = "0q") -> Optional[float]:
    """Net 30-day EPS revision breadth in [-1, 1], or None when nobody revised.

    +1 means every revision was upward. `estimates` is the payload stored by
    update_data; `horizon` keys into yfinance's eps_revisions frame.
    """
    revisions = (estimates or {}).get("eps_revisions") or {}
    row = revisions.get(horizon)
    if not row:
        return None

    up = row.get("upLast30days")
    down = row.get("downLast30days")
    if up is None or down is None:
        return None

    total = up + down
    if total <= 0:
        return None
    return (up - down) / total
