"""Unit tests for quarterly-statement trend helpers and TTM ROIC.

Pure stdlib — runnable without the app's dependencies:

    python tests/test_statement_trends.py
    pytest tests/test_statement_trends.py

The load-bearing test is `test_ttm_roic_is_on_an_annual_scale`: running the
annual per-period helper over quarterly data returns roughly a quarter of the
true ROIC, silently and without error.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.utils.roic import _roic_for_period, get_roic_ttm_series
from backend.utils.statement_trends import (
    eps_next_quarter_growth,
    eps_revision_ratio,
    gross_margin_trend,
    operating_margin_trend,
    revenue_growth_yoy_avg,
    trend,
)

QUARTERS = ["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30",
            "2025-06-30", "2025-03-31", "2024-12-31", "2024-09-30"]


def income(revenue, gross, operating, pretax=None, tax=None):
    return {
        "Total Revenue": revenue,
        "Gross Profit": gross,
        "Operating Income": operating,
        "Pretax Income": pretax if pretax is not None else operating,
        "Tax Provision": tax if tax is not None else operating * 0.2,
    }


def balance(assets, current_liabilities, cash):
    return {"Total Assets": assets, "Current Liabilities": current_liabilities,
            "Cash And Cash Equivalents": cash}


def flat_quarters(n=8, revenue=1000.0, gross=600.0, operating=250.0):
    """n identical quarters — any trend over these must be exactly 0."""
    return (
        {q: income(revenue, gross, operating) for q in QUARTERS[:n]},
        {q: balance(10_000.0, 2_000.0, 1_000.0) for q in QUARTERS[:n]},
    )


# --- TTM ROIC: the annualisation trap ---


def test_ttm_roic_is_on_an_annual_scale():
    inc, bs = flat_quarters()
    ttm = get_roic_ttm_series(bs, inc)[0]
    quarterly = _roic_for_period(bs[QUARTERS[0]], inc[QUARTERS[0]])

    # NOPAT = 250 * (1 - 0.2) = 200/quarter; invested capital = 10000-2000-1000 = 7000.
    assert abs(ttm - (800 / 7000 * 100)) < 1e-6, ttm
    # The annual helper on one quarter is ~1/4 of that — the bug this guards.
    assert abs(quarterly * 4 - ttm) < 1e-6


def test_ttm_roic_uses_average_invested_capital():
    inc, _ = flat_quarters()
    growing = dict(zip(QUARTERS[:4], [balance(c, 2_000.0, 1_000.0)
                                      for c in (16_000.0, 14_000.0, 12_000.0, 10_000.0)]))
    ttm = get_roic_ttm_series(growing, inc)[0]
    # Average invested capital = mean(13000, 11000, 9000, 7000) = 10000.
    assert abs(ttm - (800 / 10_000 * 100)) < 1e-6, ttm


def test_ttm_roic_returns_as_many_windows_as_the_data_supports():
    # Yahoo usually carries only 4-6 quarters, so asking for 4 windows must not
    # return nothing when only one window is available.
    inc4, bs4 = flat_quarters(n=4)
    assert len(get_roic_ttm_series(bs4, inc4, max_windows=4)) == 1
    inc6, bs6 = flat_quarters(n=6)
    assert len(get_roic_ttm_series(bs6, inc6, max_windows=4)) == 3
    inc8, bs8 = flat_quarters(n=8)
    assert len(get_roic_ttm_series(bs8, inc8, max_windows=4)) == 4


def test_ttm_roic_needs_a_full_four_quarters():
    inc, bs = flat_quarters(n=3)
    assert get_roic_ttm_series(bs, inc) == []
    assert get_roic_ttm_series({}, {}) == []


def test_ttm_roic_rejects_non_positive_invested_capital():
    inc, _ = flat_quarters()
    negative = {q: balance(1_000.0, 2_000.0, 500.0) for q in QUARTERS[:4]}
    assert get_roic_ttm_series(negative, inc) == []


# --- trends ---


def test_flat_quarters_have_no_trend():
    inc, _ = flat_quarters()
    assert abs(gross_margin_trend(inc)) < 1e-9
    assert abs(operating_margin_trend(inc)) < 1e-9


def test_improving_margin_is_positive():
    # Newest first: 30% -> 25% -> 20% -> 15% operating margin.
    inc = {q: income(1000.0, 600.0, m) for q, m in zip(QUARTERS[:4], [300.0, 250.0, 200.0, 150.0])}
    assert abs(operating_margin_trend(inc) - 15.0) < 1e-9


def test_deteriorating_margin_is_negative():
    inc = {q: income(1000.0, 600.0, m) for q, m in zip(QUARTERS[:4], [150.0, 200.0, 250.0, 300.0])}
    assert operating_margin_trend(inc) < 0


def test_trend_returns_none_rather_than_zero_when_too_short():
    # None means "no direction known"; 0 would read as "flat".
    assert trend([]) is None
    assert trend([5.0]) is None
    assert trend([5.0, 5.0]) == 0.0


def test_margin_trend_needs_four_computable_quarters():
    inc = {q: income(1000.0, 600.0, 250.0) for q in QUARTERS[:3]}
    assert gross_margin_trend(inc) is None
    missing_revenue = {q: {"Gross Profit": 600.0} for q in QUARTERS[:4]}
    assert gross_margin_trend(missing_revenue) is None


# --- revenue growth ---


def test_revenue_growth_averages_four_yoy_comparisons():
    # Every quarter is 20% above the same quarter a year earlier.
    inc = {}
    for i, q in enumerate(QUARTERS[:4]):
        inc[q] = income(1200.0, 600.0, 250.0)
    for q in QUARTERS[4:8]:
        inc[q] = income(1000.0, 500.0, 200.0)
    assert abs(revenue_growth_yoy_avg(inc) - 20.0) < 1e-9


def test_revenue_growth_uses_however_many_pairs_exist():
    # Yahoo often carries 4-6 quarters: 6 gives 2 YoY pairs, which is the minimum.
    inc6, _ = flat_quarters(n=6)
    assert revenue_growth_yoy_avg(inc6) == 0.0
    inc8, _ = flat_quarters(n=8)
    assert revenue_growth_yoy_avg(inc8) == 0.0


def test_revenue_growth_refuses_a_single_pair():
    # One pair would just restate `revenue_growth`, which already exists.
    inc5, _ = flat_quarters(n=5)
    assert revenue_growth_yoy_avg(inc5) is None
    inc4, _ = flat_quarters(n=4)
    assert revenue_growth_yoy_avg(inc4) is None


def test_revenue_growth_smooths_a_single_spike():
    # One 300% quarter against three flat ones must not read as 300% growth.
    inc = {QUARTERS[0]: income(4000.0, 600.0, 250.0)}
    for q in QUARTERS[1:4]:
        inc[q] = income(1000.0, 600.0, 250.0)
    for q in QUARTERS[4:8]:
        inc[q] = income(1000.0, 500.0, 200.0)
    avg = revenue_growth_yoy_avg(inc)
    assert 70.0 < avg < 80.0, avg


# --- estimate revisions ---


def test_revision_ratio_spans_minus_one_to_one():
    up_only = {"eps_revisions": {"0q": {"upLast30days": 8, "downLast30days": 0}}}
    down_only = {"eps_revisions": {"0q": {"upLast30days": 0, "downLast30days": 5}}}
    balanced = {"eps_revisions": {"0q": {"upLast30days": 3, "downLast30days": 3}}}
    assert eps_revision_ratio(up_only) == 1.0
    assert eps_revision_ratio(down_only) == -1.0
    assert eps_revision_ratio(balanced) == 0.0


def test_next_quarter_growth_is_a_percentage():
    estimates = {"earnings_estimate": {"+1q": {"growth": 0.185}, "0q": {"growth": 0.05}}}
    assert abs(eps_next_quarter_growth(estimates) - 18.5) < 1e-9
    assert eps_next_quarter_growth({"earnings_estimate": {"+1q": {"growth": None}}}) is None
    assert eps_next_quarter_growth({}) is None
    assert eps_next_quarter_growth(None) is None


def test_revision_ratio_is_none_when_nobody_revised():
    # Distinct from 0.0, which means revisions happened and offset.
    assert eps_revision_ratio({"eps_revisions": {"0q": {"upLast30days": 0, "downLast30days": 0}}}) is None
    assert eps_revision_ratio({}) is None
    assert eps_revision_ratio(None) is None
    assert eps_revision_ratio({"eps_revisions": {}}) is None


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
