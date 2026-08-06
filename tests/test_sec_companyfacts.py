"""Tests for SEC companyfacts vintage selection and duration filters."""

from datetime import date

from backend.utils.sec_companyfacts import FactPoint, select_vintage, _is_annual_duration, _is_quarter_duration


def _fp(end, filed, form="10-K", start=None, value=1.0, accn="a"):
    return FactPoint(
        taxonomy="us-gaap",
        concept="OperatingIncomeLoss",
        end=end,
        start=start,
        filed=filed,
        form=form,
        value=value,
        unit="USD",
        accn=accn,
        fp="FY" if form.startswith("10-K") else "Q2",
    )


def test_select_vintage_pit_is_earliest_filed():
    end = date(2023, 12, 31)
    start = date(2023, 1, 1)
    points = [
        _fp(end, date(2024, 2, 1), start=start, value=10, accn="2"),
        _fp(end, date(2024, 1, 15), start=start, value=9, accn="1"),
        _fp(end, date(2025, 2, 1), start=start, value=11, accn="3"),  # restatement
    ]
    pit = select_vintage(points, "pit")
    assert pit[(start, end)].value == 9
    restated = select_vintage(points, "as_restated")
    assert restated[(start, end)].value == 11


def test_annual_and_quarter_duration_heuristics():
    assert _is_annual_duration(date(2023, 1, 1), date(2023, 12, 31), "10-K", "FY")
    assert _is_quarter_duration(date(2023, 4, 1), date(2023, 6, 30), "10-Q", "Q2")
    assert not _is_annual_duration(date(2023, 4, 1), date(2023, 6, 30), "10-Q", "Q2")
