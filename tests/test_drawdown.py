"""Drawdown maths behind the Dashboard's Max DD figure and the underwater chart."""

from datetime import date, timedelta

import pytest

from backend.utils.drawdown import compute_max_drawdown, underwater_series
from backend.views.portfolio import _twrr_wealth_index


def test_max_drawdown_is_the_worst_fall_from_a_prior_peak():
    assert compute_max_drawdown([100, 120, 90, 130, 110]) == {
        "max_drawdown_pct": -25.0,
        "max_drawdown_abs": -30.0,
        "drawdown_duration_days": 1,
    }


def test_rising_series_has_no_drawdown():
    assert compute_max_drawdown([100, 101, 102]) == {
        "max_drawdown_pct": 0.0,
        "max_drawdown_abs": 0.0,
        "drawdown_duration_days": 0,
    }


def test_missing_values_are_skipped_but_still_count_as_days():
    result = compute_max_drawdown([100, None, 80])
    assert result["max_drawdown_pct"] == pytest.approx(-20.0)
    assert result["drawdown_duration_days"] == 2


def test_underwater_is_the_fall_from_the_running_peak():
    assert underwater_series([100, 110, 99, None, 121]) == pytest.approx([0.0, 0.0, -10.0, None, 0.0])


def test_wealth_index_starts_at_one_and_drops_unusable_returns():
    start = date(2025, 1, 1)
    rows = [(start, 12.3), (start + timedelta(days=1), None), (start + timedelta(days=2), -100.0)]
    assert _twrr_wealth_index(rows) == [1.0, None, None]
