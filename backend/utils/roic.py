"""ROIC helper utilities."""

from typing import Any, Iterator, Optional

DEFAULT_TAX_RATE = 0.25
MAX_TAX_RATE = 0.5
QUARTERS_PER_YEAR = 4


def _invested_capital(bs: dict[str, Any]) -> Optional[float]:
    """Total assets less current liabilities and cash, or debt + equity - cash."""
    cash_and_equiv = (
        bs.get("Cash And Cash Equivalents") or bs.get("Cash Cash Equivalents And Short Term Investments") or 0.0
    )

    total_assets = bs.get("Total Assets")
    current_liabilities = bs.get("Current Liabilities") or bs.get("Total Current Liabilities")
    if total_assets is not None and current_liabilities is not None:
        return total_assets - current_liabilities - cash_and_equiv

    total_debt = bs.get("Total Debt") or (bs.get("Long Term Debt", 0.0) + bs.get("Current Debt", 0.0))
    total_equity = bs.get("Stockholders Equity") or bs.get("Total Stockholders Equity")
    if total_debt is not None and total_equity is not None:
        return total_debt + total_equity - cash_and_equiv

    return None


def _roic_for_period(bs: dict[str, Any], is_stmt: dict[str, Any]) -> Optional[float]:
    """
    ROIC as a percentage for one balance-sheet / income-statement pair.

    Annual statements only — on quarterly data this returns a quarter's return,
    not an annualised one. Use `get_roic_ttm_series` for quarterly input.
    """
    nopat = _nopat(is_stmt)
    if nopat is None:
        return None

    invested_capital = _invested_capital(bs)
    if invested_capital is None or invested_capital <= 0:
        return None

    return float((nopat / invested_capital) * 100.0)


def _paired_periods(
    balance_sheet: dict[str, Any], income_stmt: dict[str, Any]
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Statement pairs newest first, matching each balance sheet to the closest prior income statement."""
    is_dates = sorted(income_stmt.keys(), reverse=True)
    for bs_date in sorted(balance_sheet.keys(), reverse=True):
        is_date = next((d for d in is_dates if d <= bs_date), None)
        if is_date is not None:
            yield balance_sheet[bs_date], income_stmt[is_date]


def _nopat(is_stmt: dict[str, Any]) -> Optional[float]:
    """After-tax operating profit for one period, or None when EBIT is missing."""
    ebit = is_stmt.get("Operating Income") or is_stmt.get("EBIT")
    if ebit is None:
        return None

    tax_expense = is_stmt.get("Tax Provision") or is_stmt.get("Income Tax Expense") or 0.0
    ebt = is_stmt.get("Pretax Income") or is_stmt.get("Income Before Tax")
    tax_rate = DEFAULT_TAX_RATE
    if ebt and ebt > 0 and tax_expense > 0:
        tax_rate = max(0.0, min(MAX_TAX_RATE, tax_expense / ebt))

    return ebit * (1.0 - tax_rate)


def get_roic_ttm_series(
    quarterly_balance_sheet: Optional[dict[str, Any]],
    quarterly_income_stmt: Optional[dict[str, Any]],
    max_windows: int = 4,
) -> list[float]:
    """
    Trailing-twelve-month ROIC percentages, newest first, one per rolling window.

    Each value sums NOPAT over four consecutive quarters and divides by the mean
    invested capital across them, so the result is on the same annual scale as
    `get_roic` — a raw quarterly ROIC would be roughly a quarter of it.

    Returns as many complete windows as the data supports, capped at
    `max_windows`; Yahoo usually carries only 4-6 quarters. Empty when not even
    one window is computable, and stops early rather than skipping a gap, so a
    returned series is always contiguous.
    """
    if not quarterly_balance_sheet or not quarterly_income_stmt:
        return []

    try:
        pairs = list(_paired_periods(quarterly_balance_sheet, quarterly_income_stmt))
    except (TypeError, ValueError, KeyError):
        return []

    series: list[float] = []
    for start in range(max_windows):
        window = pairs[start : start + QUARTERS_PER_YEAR]
        if len(window) < QUARTERS_PER_YEAR:
            break
        nopats = [_nopat(is_stmt) for _, is_stmt in window]
        capitals = [_invested_capital(bs) for bs, _ in window]
        if any(n is None for n in nopats) or any(c is None or c <= 0 for c in capitals):
            break
        avg_capital = sum(capitals) / len(capitals)  # type: ignore[arg-type]
        series.append(float(sum(nopats) / avg_capital * 100.0))  # type: ignore[arg-type]

    return series


def get_roic_history(
    balance_sheet: Optional[dict[str, Any]],
    income_stmt: Optional[dict[str, Any]],
    periods: int = 3,
) -> list[float]:
    """ROIC percentages for the most recent `periods` fiscal years, newest first.

    Empty unless every one of those periods is computable — a partial run would
    let a 2-year sample pass a 3-year consistency test.
    """
    if not balance_sheet or not income_stmt:
        return []

    history: list[float] = []
    try:
        for bs, is_stmt in _paired_periods(balance_sheet, income_stmt):
            if len(history) == periods:
                break
            roic = _roic_for_period(bs, is_stmt)
            if roic is None:
                return []
            history.append(roic)
    except (TypeError, ValueError, KeyError, ZeroDivisionError):
        return []

    return history if len(history) == periods else []


def get_roic(
    info: dict[str, Any],
    balance_sheet: Optional[dict[str, Any]] = None,
    income_stmt: Optional[dict[str, Any]] = None,
) -> Optional[float]:
    """Calculate Return on Invested Capital (ROIC) from statements or info proxy.

    Prefers statement-based calculation (NOPAT / Invested Capital) using
    balance_sheet and income_stmt. Falls back to info-based proxy if missing.
    """
    if balance_sheet and income_stmt:
        try:
            for bs, is_stmt in _paired_periods(balance_sheet, income_stmt):
                roic = _roic_for_period(bs, is_stmt)
                if roic is not None:
                    return roic
                break
        except (TypeError, ValueError, KeyError, ZeroDivisionError):
            pass  # Fall back to info-based proxy if statements are malformed

    roic = None
    try:
        total_debt = info.get("totalDebt")
        total_cash = info.get("totalCash")
        debt_to_equity_ratio = info.get("debtToEquity")  # percentage, e.g. 85.3
        ebit = info.get("ebitda")  # Using EBITDA as proxy for EBIT

        if total_debt is not None and total_cash is not None and debt_to_equity_ratio is not None and ebit is not None:
            total_equity = (total_debt / (debt_to_equity_ratio / 100.0)) if debt_to_equity_ratio > 0 else 0
            invested_capital = total_debt + total_equity - total_cash

            nopat = ebit * (1 - DEFAULT_TAX_RATE)

            if invested_capital > 0:
                roic = (nopat / invested_capital) * 100.0

    except (TypeError, ZeroDivisionError):
        roic = None

    return roic
