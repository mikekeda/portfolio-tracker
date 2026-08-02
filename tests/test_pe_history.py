"""Unit tests for the historical P/E helpers.

Pure stdlib — runnable without the app's dependencies:

    python tests/test_pe_history.py
    pytest tests/test_pe_history.py

The load-bearing test is `test_basis_matches_ignores_window_width`: `basis_matches`
takes a parsed series so callers can compute every P/E statistic from one parse,
which is only safe while a wider window gives the same verdict as a narrow one.
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.utils.pe_history import (
    MAX_BASIS_DIVERGENCE,
    MIN_PE_SAMPLE,
    avg_pe,
    basis_matches,
    harmonic_mean_pe,
    pe_series,
)

TODAY = date(2026, 8, 2)


def pes(*pairs):
    return {day: {"pe_ratio": pe} for day, pe in pairs}


def quarterly(count, pe, start_year=2026, start_month=6):
    """`count` quarter-ends counting backwards from start, all at the same P/E."""
    out = {}
    year, month = start_year, start_month
    for _ in range(count):
        out[date(year, month, 30).isoformat()] = {"pe_ratio": pe}
        month -= 3
        if month <= 0:
            month += 12
            year -= 1
    return out


def test_pe_series_excludes_forward_estimates_and_stale_points():
    data = pes(("2026-12-31", 40.0), ("2026-06-30", 20.0), ("2019-06-30", 10.0))
    got = pe_series(data, TODAY, years=5)
    # The 2026-12-31 key is a Wisesheets forward estimate; 2019 is outside 5y.
    assert [d.isoformat() for d, _ in got] == ["2026-06-30"], got


def test_pe_series_drops_non_positive_and_sorts_oldest_first():
    data = pes(("2026-06-30", 20.0), ("2025-06-30", -5.0), ("2024-06-30", 10.0))
    assert [pe for _, pe in pe_series(data, TODAY, years=5)] == [10.0, 20.0]


def test_harmonic_mean_is_not_dragged_by_a_trough_quarter():
    # A single 400x quarter would pull an arithmetic mean to ~85; the harmonic
    # mean of the earnings yields stays near the typical multiple.
    values = [10.0] * (MIN_PE_SAMPLE - 1) + [400.0]
    got = harmonic_mean_pe(values)
    assert got is not None and got < 12.0, got
    assert sum(values) / len(values) > 45.0


def test_harmonic_mean_needs_a_minimum_sample():
    assert harmonic_mean_pe([10.0] * (MIN_PE_SAMPLE - 1)) is None
    assert harmonic_mean_pe([]) is None
    assert harmonic_mean_pe([10.0] * MIN_PE_SAMPLE) is not None


def test_avg_pe_matches_harmonic_mean_of_the_series():
    data = quarterly(MIN_PE_SAMPLE, 25.0)
    assert abs(avg_pe(data, TODAY) - 25.0) < 1e-9
    assert avg_pe({}, TODAY) is None


def test_basis_matches_accepts_a_comparable_current_pe():
    data = quarterly(4, 20.0)
    series = pe_series(data, TODAY, years=5)
    assert basis_matches(series, TODAY, 20.0 * (1 + MAX_BASIS_DIVERGENCE / 2))
    assert not basis_matches(series, TODAY, 20.0 * (1 + MAX_BASIS_DIVERGENCE * 2))


def test_basis_matches_is_false_without_a_recent_scraped_point():
    # Newest point predates BASIS_LOOKBACK_MONTHS, so there is nothing to compare.
    series = pe_series(quarterly(4, 20.0, start_year=2024), TODAY, years=5)
    assert not basis_matches(series, TODAY, 20.0)


def test_basis_matches_ignores_window_width():
    # Callers pass a 5y series to avoid reparsing; it must agree with the 1y
    # window the helper was originally written against.
    data = quarterly(20, 20.0)
    wide = pe_series(data, TODAY, years=5)
    narrow = pe_series(data, TODAY, years=1)
    for current_pe in (18.0, 20.0, 26.0, 40.0):
        assert basis_matches(wide, TODAY, current_pe) == basis_matches(narrow, TODAY, current_pe), current_pe


def test_leap_day_does_not_crash_the_window_shift():
    data = pes(("2024-02-29", 15.0))
    # Shifting 5y off a 29 Feb `today` has no same-day counterpart; it clamps to
    # the 28th rather than raising, and the window still spans the point.
    assert [pe for _, pe in pe_series(data, date(2028, 2, 29), years=5)] == [15.0]
    assert [pe for _, pe in pe_series(data, date(2029, 2, 28), years=5)] == [15.0]
    assert pe_series(data, date(2029, 3, 1), years=5) == []


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
