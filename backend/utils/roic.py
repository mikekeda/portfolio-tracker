from typing import Any, Optional


def get_roic(info: dict[str, Any]) -> Optional[float]:
    roic = None
    try:
        # 1. Get components from yfinance info
        total_debt = info.get("totalDebt")
        total_cash = info.get("totalCash")
        debt_to_equity_ratio = info.get("debtToEquity")  # This is a percentage, e.g., 85.3
        ebit = info.get("ebitda")  # Using EBITDA as a proxy for EBIT

        if total_debt is not None and total_cash is not None and debt_to_equity_ratio is not None and ebit is not None:
            # 2. Derive Total Equity and calculate Invested Capital
            total_equity = (total_debt / (debt_to_equity_ratio / 100.0)) if debt_to_equity_ratio > 0 else 0
            invested_capital = total_debt + total_equity - total_cash

            # 3. Calculate NOPAT (using an estimated 25% tax rate)
            tax_rate = 0.25
            nopat = ebit * (1 - tax_rate)

            # 4. Calculate final ROIC percentage
            if invested_capital > 0:
                roic = (nopat / invested_capital) * 100.0

    except (TypeError, ZeroDivisionError):
        roic = None  # Handle cases where data is missing or division by zero occurs

    return roic
