from backend.utils.piotroski import get_piotroski_f_score


def test_piotroski_uses_alias_for_both_cashflow_checks():
    income = {d: {"Net Income": 50, "Total Revenue": 300, "Gross Profit": 100} for d in ["2025-12-31", "2024-12-31"]}
    bs = {
        d: {
            "Total Assets": 500,
            "Current Assets": 150,
            "Current Liabilities": 100,
            "Long Term Debt": 50,
            "Ordinary Shares Number": 100,
        }
        for d in income
    }
    cf = {"2025-12-31": {"Cash Flow From Continuing Operating Activities": 120}}
    result = get_piotroski_f_score(cf, bs, income)
    assert result["available"] == 9
    assert result["details"]["Operating cash flow > 0"]
    assert result["details"]["OCF > net income (quality)"]
