"""Unit tests for cumulative split-factor helpers."""

from datetime import date

from backend.utils.splits import adjust_share_count, cumulative_split_factor


def test_cumulative_factor_product_of_later_splits():
    splits = {
        "2021-07-20": 4.0,
        "2024-06-10": 10.0,
        "2007-09-11": 1.5,
    }
    # Before 2007: all three apply (splits after as_of)
    assert cumulative_split_factor(splits, date(2000, 1, 1)) == 1.5 * 4.0 * 10.0
    # After 2007, before 2021: 4x and 10x
    assert cumulative_split_factor(splits, date(2020, 1, 1)) == 4.0 * 10.0
    # Between 2021 and 2024: only 10x
    assert cumulative_split_factor(splits, date(2022, 1, 1)) == 10.0
    # On/after last split: identity (date > as_of, so equality excluded)
    assert cumulative_split_factor(splits, date(2024, 6, 10)) == 1.0
    assert cumulative_split_factor(splits, date(2025, 1, 1)) == 1.0


def test_float_and_reverse_split_factors():
    splits = {"2014-04-03": 1.998, "2003-07-01": 0.5}
    assert abs(cumulative_split_factor(splits, date(2000, 1, 1)) - 1.998 * 0.5) < 1e-9
    assert abs(cumulative_split_factor(splits, date(2010, 1, 1)) - 1.998) < 1e-9


def test_adjust_share_count_none_safe():
    assert adjust_share_count(None, {"2024-06-10": 10.0}, date(2020, 1, 1)) is None
    assert adjust_share_count(100.0, None, date(2020, 1, 1)) == 100.0
    assert adjust_share_count(100.0, {"2024-06-10": 10.0}, date(2020, 1, 1)) == 1000.0
