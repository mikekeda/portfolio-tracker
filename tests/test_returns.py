"""MWRR/TWRR maths behind the Dashboard, drawdown, Projection and Risk figures."""

from datetime import date, timedelta

import pytest

from backend.views.portfolio import _twrr_wealth_index
from config import DAYS_PER_YEAR
from scripts.update_returns import _daily_irr, compute_returns

START = date(2025, 1, 1)


def day(n: int) -> date:
    return START + timedelta(days=n)


# +10% on the first £100, then a second £100 arrives just before a 10% fall.
SNAPSHOTS = [(day(0), 100.0), (day(181), 110.0), (day(182), 210.0), (day(365), 189.0)]
FLOWS = {day(0): 100.0, day(182): 100.0}


def cumulative(annual_pct: float, days: int) -> float:
    """Growth factor over `days` implied by an annualised percentage return."""
    return (1 + annual_pct / 100) ** (days / DAYS_PER_YEAR)


def test_irr_of_a_single_period():
    assert _daily_irr([-100, 110]) == pytest.approx(0.10)
    assert _daily_irr([-100, 90]) == pytest.approx(-0.10)


def test_irr_compounds_across_periods():
    assert _daily_irr([-100, 0, 121]) == pytest.approx(0.10)


def test_irr_is_zero_when_flows_net_to_zero():
    assert _daily_irr([-100, 40, 60]) == 0.0


@pytest.mark.parametrize("flows", [[100, 100], [-100, -50]])
def test_irr_is_none_without_a_sign_change(flows):
    assert _daily_irr(flows) is None


def test_twrr_chains_period_returns_and_ignores_deposits():
    mwrr, twrr = compute_returns(SNAPSHOTS, FLOWS)[-1][1:]
    assert cumulative(twrr, 365) == pytest.approx(1.1 * 0.9)
    # More money was exposed to the fall than to the gain.
    assert mwrr < twrr


def test_mwrr_is_the_irr_of_deposits_and_final_value():
    mwrr = compute_returns(SNAPSHOTS, FLOWS)[-1][1]
    daily = (1 + mwrr / 100) ** (1 / DAYS_PER_YEAR) - 1
    npv = -100 - 100 * (1 + daily) ** -182 + 189 * (1 + daily) ** -365
    assert npv == pytest.approx(0, abs=1e-6)


def test_flows_without_market_moves_score_zero():
    snapshots = [(day(0), 100.0), (day(100), 150.0), (day(200), 110.0)]
    flows = {day(0): 100.0, day(100): 50.0, day(200): -40.0}
    for _, mwrr, twrr in compute_returns(snapshots, flows):
        assert (mwrr, twrr) == (0.0, 0.0)


def test_deposit_between_snapshots_is_capital_not_return():
    snapshots = [(day(0), 100.0), (day(10), 210.0)]
    mwrr, twrr = compute_returns(snapshots, {day(0): 100.0, day(5): 100.0})[-1][1:]
    assert cumulative(twrr, 10) == pytest.approx(1.05)
    # MWRR still dates the deposit on its own day.
    daily = (1 + mwrr / 100) ** (1 / DAYS_PER_YEAR) - 1
    assert -100 - 100 * (1 + daily) ** -5 + 210 * (1 + daily) ** -10 == pytest.approx(0, abs=1e-6)


def test_deposit_before_the_first_snapshot_does_not_move_twrr():
    snapshots = [(day(2), 100.0), (day(367), 110.0)]
    same_day = compute_returns(snapshots, {day(2): 100.0})[-1]
    earlier = compute_returns(snapshots, {day(0): 100.0})[-1]
    assert earlier[2] == pytest.approx(same_day[2])
    # The same gain on money held two days longer is a lower IRR.
    assert earlier[1] < same_day[1]


def test_target_dates_only_skip_work():
    targets = {day(182), day(365)}
    full = compute_returns(SNAPSHOTS, FLOWS)
    assert compute_returns(SNAPSHOTS, FLOWS, target_dates=targets) == [row for row in full if row[0] in targets]


def test_no_snapshots_no_returns():
    assert compute_returns([], FLOWS) == []


def test_dashboard_wealth_index_recovers_the_chained_returns():
    rows = [(d, twrr) for d, _, twrr in compute_returns(SNAPSHOTS, FLOWS)]
    assert _twrr_wealth_index(rows) == pytest.approx([1.0, 1.1, 1.1, 0.99])
