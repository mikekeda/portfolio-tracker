"""Free cash flow and its matching revenue, read from cached Yahoo statements.

Yahoo's `info["freeCashflow"]` is a *levered* figure — net of debt service — and
understates statement FCF by more than 30% for 21% of the universe (MSFT 25%,
GOOGL 31%), so margins and yields built on it are wrong. Both figures returned
here span the same periods and are in the reporting currency, which makes the
ratio FX-immune.

Stock-based compensation is deliberately not deducted: these feed quality gates
calibrated on conventional unlevered FCF margin. `utils/dcf.py` does deduct it,
because dilution matters when discounting to a per-share value.
"""

from typing import Any, NamedTuple, Optional

QUARTERS_PER_YEAR = 4

# Yahoo labels the top line differently across statement vintages.
_REVENUE_LABELS = ("Total Revenue", "Operating Revenue")


class TrailingFcf(NamedTuple):
    """Free cash flow and revenue over the same periods, in the reporting currency."""

    fcf: float
    revenue: float


def _number(value: Any) -> Optional[float]:
    """Finite float, or None for absent, NaN, infinite or non-numeric values."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _rows(statement: Optional[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Period rows keyed by ISO date, skipping anything not shaped like a row."""
    if not isinstance(statement, dict):
        return {}
    return {period: row for period, row in statement.items() if isinstance(row, dict)}


def _period_fcf(row: dict[str, Any]) -> Optional[float]:
    """One period's FCF, derived from cash flow and capex when not reported directly."""
    reported = _number(row.get("Free Cash Flow"))
    if reported is not None:
        return reported

    operating_cashflow = _number(row.get("Operating Cash Flow"))
    capex = _number(row.get("Capital Expenditure"))
    if operating_cashflow is None or capex is None:
        return None
    return operating_cashflow + capex  # Yahoo signs capex negative


def _period_revenue(row: dict[str, Any]) -> Optional[float]:
    """One period's revenue, or None when no known label carries a number."""
    for label in _REVENUE_LABELS:
        revenue = _number(row.get(label))
        if revenue is not None:
            return revenue
    return None


def _ttm(
    quarterly_cashflow: Optional[dict[str, Any]], quarterly_income_stmt: Optional[dict[str, Any]]
) -> Optional[TrailingFcf]:
    """Sums over the four most recent quarters present in both statements.

    Both sides read the identical key set, so the ratio survives a hole in the
    quarterly history (15 names) and needs no `info` field to be consistent with
    the statements — Yahoo reports VALE and PBR revenue in BRL there while their
    statements are in USD.
    """
    cashflow_rows = _rows(quarterly_cashflow)
    income_rows = _rows(quarterly_income_stmt)

    quarters = sorted(cashflow_rows.keys() & income_rows.keys(), reverse=True)[:QUARTERS_PER_YEAR]
    if len(quarters) < QUARTERS_PER_YEAR:
        return None

    fcf = [_period_fcf(cashflow_rows[quarter]) for quarter in quarters]
    revenue = [_period_revenue(income_rows[quarter]) for quarter in quarters]
    if any(value is None for value in (*fcf, *revenue)) or sum(revenue) <= 0:
        return None
    return TrailingFcf(sum(fcf), sum(revenue))


def _annual(cashflow: Optional[dict[str, Any]], income_stmt: Optional[dict[str, Any]]) -> Optional[TrailingFcf]:
    """The most recent fiscal year computable from both statements."""
    cashflow_rows = _rows(cashflow)
    income_rows = _rows(income_stmt)

    for year in sorted(cashflow_rows.keys() & income_rows.keys(), reverse=True):
        fcf = _period_fcf(cashflow_rows[year])
        revenue = _period_revenue(income_rows[year])
        if fcf is not None and revenue is not None and revenue > 0:
            return TrailingFcf(fcf, revenue)
    return None


def trailing_fcf(
    quarterly_cashflow: Optional[dict[str, Any]],
    quarterly_income_stmt: Optional[dict[str, Any]],
    cashflow: Optional[dict[str, Any]],
    income_stmt: Optional[dict[str, Any]],
) -> Optional[TrailingFcf]:
    """Latest free cash flow and matching revenue, trailing-twelve-month where possible.

    Falls back to the most recent fiscal year when four common quarters aren't
    cached (Yahoo omits quarterly statements for ~19% of the universe, mostly
    non-US listings). None when neither basis is computable.
    """
    return _ttm(quarterly_cashflow, quarterly_income_stmt) or _annual(cashflow, income_stmt)
