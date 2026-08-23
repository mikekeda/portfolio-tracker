"""Unit tests for the ONS CPI parser and rate derivation.

Pure stdlib — runnable without the app's dependencies:

    python tests/test_cpi.py
    pytest tests/test_cpi.py

The load-bearing test is `test_parse_skips_annual_and_quarterly_rows`: the MM23
CSV interleaves annual, quarterly and monthly rows under one header, and an annual
row parsed as monthly would put a whole-year value into the 12m YoY calculation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.utils.cpi import cpi_metrics, parse_ons_csv

# Abridged MM23 export: ONS puts metadata, then annual, quarterly and monthly blocks.
MM23_CSV = '''"Title","CPI INDEX 00: ALL ITEMS 2015=100"
"CDID","D7BT"
"Source dataset ID","MM23"
"2023","130.5"
"2024","134.2"
"2025 Q1","136.0"
"2025 Q2","137.1"
"2025 JAN","135.8"
"2025 FEB","136.1"
"2025 MAR","136.4"
"2025 APR","137.0"
"2025 MAY","137.2"
"2025 JUN","137.4"
"2025 JUL","137.7"
"2025 AUG","138.0"
"2025 SEP","138.3"
"2025 OCT","138.6"
"2025 NOV","138.8"
"2025 DEC","139.1"
"2026 JAN","140.7"
'''


def monthly(count, start_index=100.0, monthly_rate=0.0):
    """`count` monthly observations ending Jan 2026, newest first.

    Values compound at `monthly_rate` forwards in time, so obs[0] is the largest.
    """
    obs = []
    for i in range(count):
        # Months before Jan 2026, as a 0-based absolute month number.
        absolute = (2026 * 12) - i
        obs.append({
            "date": f"{absolute // 12}-{absolute % 12 + 1:02d}-01",
            "value": start_index * (1 + monthly_rate) ** (count - 1 - i),
        })
    return obs


def test_the_monthly_helper_emits_distinct_descending_months():
    # Pins the fixture itself: an off-by-one here silently duplicates a month and
    # makes every rate assertion below measure the wrong span.
    dates = [o["date"] for o in monthly(14)]
    assert dates[:3] == ["2026-01-01", "2025-12-01", "2025-11-01"]
    assert dates[12] == "2025-01-01"
    assert len(set(dates)) == len(dates)
    assert dates == sorted(dates, reverse=True)


def test_parse_skips_annual_and_quarterly_rows():
    obs = parse_ons_csv(MM23_CSV)
    assert len(obs) == 13
    assert all(len(o["date"]) == 10 for o in obs)
    # 130.5 (annual 2023) and 136.0 (2025 Q1) must not appear as observations.
    assert 130.5 not in [o["value"] for o in obs]
    assert 136.0 not in [o["value"] for o in obs]


def test_parse_returns_newest_first_with_iso_dates():
    obs = parse_ons_csv(MM23_CSV)
    assert obs[0] == {"date": "2026-01-01", "value": 140.7}
    assert obs[-1] == {"date": "2025-01-01", "value": 135.8}


def test_parse_zero_pads_single_digit_months():
    # "2025-1-01" would sort after "2025-10-01" as a string and scramble the series.
    dates = [o["date"] for o in parse_ons_csv(MM23_CSV)]
    assert dates == sorted(dates, reverse=True)
    assert "2025-09-01" in dates


def test_parse_ignores_unparseable_values_and_short_rows():
    csv = '"2026 JAN","140.7"\n"2026 FEB",".."\n"2026 MAR"\n"2026 APR","x","y"\n'
    assert parse_ons_csv(csv) == [{"date": "2026-01-01", "value": 140.7}]


def test_parse_returns_none_when_no_monthly_rows():
    assert parse_ons_csv('"Title","CPI"\n"2024","134.2"\n') is None
    assert parse_ons_csv("") is None


def test_metrics_are_decimals_not_percentages():
    # The projection page multiplies by 100 to display; returning 3.6 here would
    # render 360%. This is the unit bug that already shipped once on TWRR.
    m = cpi_metrics(monthly(13, monthly_rate=1.036 ** (1 / 12) - 1))
    assert abs(m["cpi_12m"] - 0.036) < 1e-9


def test_twelve_month_rate_uses_the_thirteenth_observation():
    obs = parse_ons_csv(MM23_CSV)
    m = cpi_metrics(obs)
    assert abs(m["cpi_12m"] - (140.7 / 135.8 - 1.0)) < 1e-12
    assert m["as_of"] == "2026-01-01"


def test_ten_year_cagr_annualises_a_short_series():
    # 61 months of 0.5%/mo: the CAGR must be annualised over 5 years, not 10.
    m = cpi_metrics(monthly(61, monthly_rate=0.005))
    assert abs(m["cpi_10y"] - (1.005 ** 12 - 1)) < 1e-9


def test_ten_year_cagr_caps_the_lookback_at_120_months():
    # A longer series must not reach past 10 years for the trailing figure.
    m = cpi_metrics(monthly(200, monthly_rate=0.002))
    assert abs(m["cpi_10y"] - (1.002 ** 12 - 1)) < 1e-9


def test_metrics_are_all_none_below_thirteen_observations():
    assert cpi_metrics(monthly(12)) == {"cpi_12m": None, "cpi_10y": None, "as_of": None}
    assert cpi_metrics(None) == {"cpi_12m": None, "cpi_10y": None, "as_of": None}


def test_flat_index_gives_zero_inflation_not_none():
    m = cpi_metrics(monthly(121, monthly_rate=0.0))
    assert m["cpi_12m"] == 0.0
    assert abs(m["cpi_10y"]) < 1e-12


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
