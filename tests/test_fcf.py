"""Unit tests for the statement-derived free cash flow helpers.

Pure stdlib — runnable without the app's dependencies:

    python tests/test_fcf.py
    pytest tests/test_fcf.py

The load-bearing test is `test_ttm_reads_both_sides_from_the_same_statements`:
the whole point of this module is that FCF and its denominator come from the
same periods of the same statements, which is what keeps the ratio FX-immune
when Yahoo's `info` disagrees with the statements about currency.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.utils.fcf import trailing_fcf

QUARTERS = ("2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30")


def quarterly_cashflow(*values, label="Free Cash Flow"):
    return {quarter: {label: value} for quarter, value in zip(QUARTERS, values)}


def quarterly_income(*values):
    return {quarter: {"Total Revenue": value} for quarter, value in zip(QUARTERS, values)}


def test_ttm_sums_four_quarters():
    result = trailing_fcf(quarterly_cashflow(10, 20, 30, 40), quarterly_income(50, 50, 50, 50), {}, {})
    assert result.fcf == 100
    assert result.revenue == 200


def test_ttm_reads_both_sides_from_the_same_statements():
    # VALE and PBR report `info` revenue in BRL against USD statements, so a
    # denominator from outside the statements would divide USD by BRL.
    result = trailing_fcf(quarterly_cashflow(10, 20, 30, 40), quarterly_income(25, 25, 25, 25), {}, {})
    assert (result.fcf, result.revenue) == (100, 100)


def test_ttm_ratio_survives_a_hole_in_the_quarterly_history():
    # 15 names skip a quarter, so the newest four common keys span 15 months.
    # Both sides read that key set, so the ratio still spans four real quarters.
    gapped = ("2026-06-30", "2026-03-31", "2025-09-30", "2025-06-30")
    cashflow = {quarter: {"Free Cash Flow": 25.0} for quarter in gapped}
    income = {quarter: {"Total Revenue": 100.0} for quarter in gapped}
    result = trailing_fcf(cashflow, income, {}, {})
    assert result.fcf / result.revenue == 0.25


def test_ttm_derives_fcf_from_operating_cashflow_and_capex():
    cashflow = {
        quarter: {"Operating Cash Flow": 100.0, "Capital Expenditure": -30.0} for quarter in QUARTERS
    }
    result = trailing_fcf(cashflow, quarterly_income(*[250] * 4), {}, {})
    assert result.fcf == 280.0


def test_ttm_needs_all_four_quarters():
    three = {quarter: {"Free Cash Flow": 10} for quarter in QUARTERS[:3]}
    annual = {"2025-12-31": {"Free Cash Flow": 99.0}}
    result = trailing_fcf(three, quarterly_income(*[250] * 4), annual,
                          {"2025-12-31": {"Total Revenue": 500.0}})
    assert result.fcf == 99.0  # fell through to the annual basis


def test_ttm_rejects_a_gap_in_the_series():
    # A missing line item in one quarter would otherwise sum three quarters of
    # cash flow against four quarters of revenue.
    holed = quarterly_cashflow(10, 20, 30, 40)
    del holed[QUARTERS[1]]["Free Cash Flow"]
    assert trailing_fcf(holed, quarterly_income(*[250] * 4), {}, {}) is None


def test_annual_pairs_its_own_revenue():
    cashflow = {"2025-12-31": {"Free Cash Flow": 60.0}, "2024-12-31": {"Free Cash Flow": 40.0}}
    income = {"2025-12-31": {"Total Revenue": 300.0}, "2024-12-31": {"Total Revenue": 200.0}}
    result = trailing_fcf({}, {}, cashflow, income)
    assert (result.fcf, result.revenue) == (60.0, 300.0)


def test_annual_skips_a_year_missing_from_either_statement():
    cashflow = {"2025-12-31": {"Free Cash Flow": 60.0}, "2024-12-31": {"Free Cash Flow": 40.0}}
    income = {"2024-12-31": {"Total Revenue": 200.0}}
    result = trailing_fcf({}, {}, cashflow, income)
    assert (result.fcf, result.revenue) == (40.0, 200.0)


def test_operating_revenue_label_is_accepted():
    income = {"2025-12-31": {"Operating Revenue": 200.0}}
    cashflow = {"2025-12-31": {"Free Cash Flow": 20.0}}
    assert trailing_fcf({}, {}, cashflow, income).revenue == 200.0


def test_negative_fcf_is_preserved():
    # A cash-burning company must read negative, not None — red_flag_cash_burn
    # keys off the sign.
    result = trailing_fcf(quarterly_cashflow(-50, -50, -50, -50), quarterly_income(*[250] * 4), {}, {})
    assert result.fcf == -200


def test_non_numeric_and_missing_statements_return_none():
    assert trailing_fcf(None, None, None, None) is None
    assert trailing_fcf({}, {}, {}, {}) is None
    assert trailing_fcf({"2025-12-31": "not a row"}, {}, {"2025-12-31": {"Free Cash Flow": "n/a"}},
                        {"2025-12-31": {"Total Revenue": 10.0}}) is None


def test_non_positive_revenue_is_not_a_denominator():
    # Both bases must agree here: `if revenue` alone would accept -1.
    for revenue in (0.0, -1.0):
        cashflow = {"2025-12-31": {"Free Cash Flow": 20.0}}
        income = {"2025-12-31": {"Total Revenue": revenue}}
        assert trailing_fcf({}, {}, cashflow, income) is None
        assert trailing_fcf(quarterly_cashflow(*[5] * 4), quarterly_income(*[revenue] * 4), {}, {}) is None


def test_booleans_are_not_numbers():
    # bool is an int subclass; True would otherwise sum as 1.0.
    cashflow = {"2025-12-31": {"Free Cash Flow": True}}
    income = {"2025-12-31": {"Total Revenue": 100.0}}
    assert trailing_fcf({}, {}, cashflow, income) is None


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
