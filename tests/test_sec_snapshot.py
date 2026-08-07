"""Regression tests for SEC snapshot PIT boundaries and thin-filer safety."""

from datetime import date

from backend.utils.sec_companyfacts import (
    FactPoint,
    compute_sec_feature_snapshot,
    concept_series,
    share_count_for_market_cap,
)


def _facts_blob(points_by_concept: dict[str, list[dict]]) -> dict:
    """Minimal companyfacts-shaped dict for unit tests."""
    us_gaap = {}
    dei = {}
    for concept, rows in points_by_concept.items():
        target = dei if concept.startswith("Entity") else us_gaap
        target[concept] = {
            "units": {
                "USD" if not concept.startswith("Entity") and "Share" not in concept else "shares": rows
            }
        }
        if "Share" in concept or concept.startswith("Entity"):
            target[concept] = {"units": {"shares": rows}}
    return {"facts": {"us-gaap": us_gaap, "dei": dei}}


def test_snapshot_survives_thin_quarterly_history():
    """Fewer than 5 quarters must not UnboundLocalError on trend columns."""
    facts = _facts_blob(
        {
            "Assets": [
                {"end": "2023-12-31", "filed": "2024-02-01", "form": "10-K", "fp": "FY", "val": 1000, "accn": "1"},
            ],
            "LiabilitiesCurrent": [
                {"end": "2023-12-31", "filed": "2024-02-01", "form": "10-K", "fp": "FY", "val": 200, "accn": "1"},
            ],
            "CashAndCashEquivalentsAtCarryingValue": [
                {"end": "2023-12-31", "filed": "2024-02-01", "form": "10-K", "fp": "FY", "val": 50, "accn": "1"},
            ],
            "OperatingIncomeLoss": [
                {
                    "end": "2023-12-31",
                    "start": "2023-01-01",
                    "filed": "2024-02-01",
                    "form": "10-K",
                    "fp": "FY",
                    "val": 100,
                    "accn": "1",
                },
            ],
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": [
                {
                    "end": "2023-12-31",
                    "start": "2023-01-01",
                    "filed": "2024-02-01",
                    "form": "10-K",
                    "fp": "FY",
                    "val": 90,
                    "accn": "1",
                },
            ],
            "IncomeTaxExpenseBenefit": [
                {
                    "end": "2023-12-31",
                    "start": "2023-01-01",
                    "filed": "2024-02-01",
                    "form": "10-K",
                    "fp": "FY",
                    "val": 20,
                    "accn": "1",
                },
            ],
        }
    )
    snap = compute_sec_feature_snapshot(
        facts, as_of=date(2024, 3, 1), splits=None, adj_close=10.0, vintage="pit"
    )
    assert snap is not None
    assert snap["gross_margin_trend_4q"] is None
    assert snap["operating_margin_trend_4q"] is None
    assert snap["roic"] is not None


def test_share_count_respects_filed_before_as_of():
    facts = _facts_blob(
        {
            "CommonStockSharesOutstanding": [
                {
                    "end": "2023-12-31",
                    "filed": "2024-02-01",
                    "form": "10-K",
                    "fp": "FY",
                    "val": 100,
                    "accn": "1",
                },
                {
                    "end": "2024-03-15",
                    "filed": "2024-03-20",
                    "form": "10-K",
                    "fp": "FY",
                    "val": 999,
                    "accn": "2",
                },
            ],
        }
    )
    # Cover-date candidate filed after as_of must be excluded.
    sc = share_count_for_market_cap(
        facts, vintage="pit", period_end=date(2023, 12, 31), as_of=date(2024, 3, 1)
    )
    assert sc is not None
    assert sc.value == 100
    assert sc.ref_date == date(2023, 12, 31)


def test_bs_share_count_ref_date_is_count_end():
    """BS cover-date after period_end must key the split factor to point.end."""
    facts = _facts_blob(
        {
            "CommonStockSharesOutstanding": [
                {
                    "end": "2024-02-15",
                    "filed": "2024-02-20",
                    "form": "10-K",
                    "fp": "FY",
                    "val": 500,
                    "accn": "1",
                },
            ],
        }
    )
    sc = share_count_for_market_cap(
        facts, vintage="pit", period_end=date(2023, 12, 31), as_of=date(2024, 3, 1)
    )
    assert sc is not None
    assert sc.value == 500
    assert sc.ref_date == date(2024, 2, 15)


def test_revenue_era_prefers_asc606_only_after_valid_from():
    facts = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "end": "2019-12-31",
                                "start": "2019-01-01",
                                "filed": "2020-02-01",
                                "form": "10-K",
                                "fp": "FY",
                                "val": 200,
                                "accn": "2",
                            },
                        ]
                    }
                },
                "SalesRevenueNet": {
                    "units": {
                        "USD": [
                            {
                                "end": "2017-12-31",
                                "start": "2017-01-01",
                                "filed": "2018-02-01",
                                "form": "10-K",
                                "fp": "FY",
                                "val": 100,
                                "accn": "1",
                            },
                        ]
                    }
                },
            }
        }
    }
    series = concept_series(facts, "revenue", vintage="as_restated", annual=True)
    assert date(2017, 12, 31) in series
    assert series[date(2017, 12, 31)].value == 100
    assert date(2019, 12, 31) in series
    assert series[date(2019, 12, 31)].value == 200


def test_snapshot_refuses_a_period_older_than_the_age_bound():
    """Financials stop tagging OperatingIncomeLoss; don't report an ancient year as current."""
    facts = _facts_blob(
        {
            "Assets": [
                {"end": "2012-12-31", "filed": "2013-02-01", "form": "10-K", "fp": "FY", "val": 1000, "accn": "1"},
            ],
            "LiabilitiesCurrent": [
                {"end": "2012-12-31", "filed": "2013-02-01", "form": "10-K", "fp": "FY", "val": 200, "accn": "1"},
            ],
            "OperatingIncomeLoss": [
                {
                    "end": "2012-12-31",
                    "start": "2012-01-01",
                    "filed": "2013-02-01",
                    "form": "10-K",
                    "fp": "FY",
                    "val": 100,
                    "accn": "1",
                },
            ],
        }
    )
    # Shortly after the filing the period is fresh enough to use.
    assert compute_sec_feature_snapshot(
        facts, as_of=date(2013, 3, 1), splits=None, adj_close=10.0, vintage="pit"
    ) is not None
    # Years later it must be None, not a stale ROIC presented as current.
    assert compute_sec_feature_snapshot(
        facts, as_of=date(2026, 7, 31), splits=None, adj_close=10.0, vintage="pit"
    ) is None
