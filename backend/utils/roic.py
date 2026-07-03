"""ROIC helper utilities."""

from typing import Any, Optional


def get_roic(
    info: dict[str, Any],
    balance_sheet: Optional[dict[str, Any]] = None,
    income_stmt: Optional[dict[str, Any]] = None,
) -> Optional[float]:
    """Calculate Return on Invested Capital (ROIC) from statements or info proxy.

    Prefers statement-based calculation (NOPAT / Invested Capital) using
    balance_sheet and income_stmt. Falls back to info-based proxy if missing.
    """
    # 1. Try statement-based calculation
    if balance_sheet and income_stmt:
        try:
            # Get latest period date
            dates = sorted(balance_sheet.keys(), reverse=True)
            if dates:
                latest_date = dates[0]
                bs = balance_sheet[latest_date]

                # Find matching or closest prior period in income statement
                is_dates = sorted(income_stmt.keys(), reverse=True)
                closest_is_date = next((d for d in is_dates if d <= latest_date), None)

                if closest_is_date:
                    is_stmt = income_stmt[closest_is_date]

                    # Extract EBIT and taxes
                    ebit = is_stmt.get("Operating Income") or is_stmt.get("EBIT")
                    tax_expense = is_stmt.get("Tax Provision") or is_stmt.get("Income Tax Expense") or 0.0
                    ebt = is_stmt.get("Pretax Income") or is_stmt.get("Income Before Tax")

                    if ebit is not None:
                        tax_rate = 0.25  # default/fallback tax rate
                        if ebt and ebt > 0 and tax_expense > 0:
                            tax_rate = tax_expense / ebt
                            tax_rate = max(0.0, min(0.5, tax_rate))

                        nopat = ebit * (1.0 - tax_rate)

                        # Calculate Invested Capital (Total Assets - Current Liabilities - Cash)
                        total_assets = bs.get("Total Assets")
                        current_liabilities = bs.get("Current Liabilities") or bs.get("Total Current Liabilities")
                        cash_and_equiv = (
                            bs.get("Cash And Cash Equivalents")
                            or bs.get("Cash Cash Equivalents And Short Term Investments")
                            or 0.0
                        )

                        invested_capital = None
                        if total_assets is not None and current_liabilities is not None:
                            invested_capital = total_assets - current_liabilities - cash_and_equiv
                        else:
                            # Fallback: Debt + Equity - Cash
                            total_debt = bs.get("Total Debt") or (
                                bs.get("Long Term Debt", 0.0) + bs.get("Current Debt", 0.0)
                            )
                            total_equity = bs.get("Stockholders Equity") or bs.get("Total Stockholders Equity")
                            if total_debt is not None and total_equity is not None:
                                invested_capital = total_debt + total_equity - cash_and_equiv

                        if invested_capital is not None and invested_capital > 0:
                            return float((nopat / invested_capital) * 100.0)
        except (TypeError, ValueError, KeyError, ZeroDivisionError):
            pass  # Fall back to info-based proxy if statements are malformed

    # 2. Fallback: Info-based proxy
    roic = None
    try:
        total_debt = info.get("totalDebt")
        total_cash = info.get("totalCash")
        debt_to_equity_ratio = info.get("debtToEquity")  # percentage, e.g. 85.3
        ebit = info.get("ebitda")  # Using EBITDA as proxy for EBIT

        if total_debt is not None and total_cash is not None and debt_to_equity_ratio is not None and ebit is not None:
            total_equity = (total_debt / (debt_to_equity_ratio / 100.0)) if debt_to_equity_ratio > 0 else 0
            invested_capital = total_debt + total_equity - total_cash

            tax_rate = 0.25
            nopat = ebit * (1 - tax_rate)

            if invested_capital > 0:
                roic = (nopat / invested_capital) * 100.0

    except (TypeError, ZeroDivisionError):
        roic = None

    return roic
