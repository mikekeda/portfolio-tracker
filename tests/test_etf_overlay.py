"""Precedence rules for the ETF look-through overlay.

The overlay fills gaps; it must never overwrite a figure the fund itself
reports, or the Holdings row and the Stock page disagree about the same fund.
Needs the venv (imports SQLAlchemy models):

    python tests/test_etf_overlay.py
    pytest tests/test_etf_overlay.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.screener_config import SCORE_NORMALIZER
from backend.views._etf_overlay import (
    FUNDAMENTALS_FIELDS,
    apply_derived_metrics,
    apply_derived_to_instrument,
)

# XNAS.L: Yahoo reports a real trailingPE of 30.55 for the fund; our constituent
# aggregate says 31.2. UKDV.L is the case where the two visibly disagree.
PAYLOAD = {
    "metrics": {
        "pe_ratio": 31.19,
        "peg_ratio": 0.85,
        "profit_margins": 18.37,
        "return_on_equity": 53.81,
        "free_cashflow_yield": 1.61,
        "roic": 27.9,
        "recommendation_mean": 1.6937,
        "form13f_score": 0.8,
        "screener_ratio": 0.427,
    },
    "coverage": {"pe_ratio": 92.5},
    "n_resolved": 98,
    "as_of": "2026-08-21",
    "distribution_yield": 0.0,
}


def test_a_fund_reported_figure_is_not_overwritten():
    holding = {"pe_ratio": 30.55, "roic": None}
    apply_derived_metrics(holding, PAYLOAD)

    assert holding["pe_ratio"] == 30.55, "Yahoo's own fund P/E must win"
    assert holding["roic"] == 27.9, "but a gap is still filled"


def test_a_missing_figure_is_filled():
    # VUAG.L and R2SC.L have no Yahoo trailingPE at all.
    holding = {"pe_ratio": None}
    apply_derived_metrics(holding, PAYLOAD)
    assert holding["pe_ratio"] == 31.19


def test_the_screener_pair_is_always_written():
    # Funds are blanked by calculate_screener_results, so there is no fund-reported
    # counterpart to defer to and the pair must land regardless.
    holding = {"screener_score": None, "screener_score_max": None}
    apply_derived_metrics(holding, PAYLOAD)

    assert holding["screener_score"] == round(0.427 * SCORE_NORMALIZER, 2)
    assert holding["screener_score_max"] == SCORE_NORMALIZER


def test_13f_lands_on_its_own_key_and_never_on_form13f_score():
    holding = {"form13f_score": None}
    apply_derived_metrics(holding, PAYLOAD)

    assert holding["look_through_form13f"] == 0.8
    assert holding["form13f_score"] is None, "filling it would claim managers hold the fund"


def test_provenance_is_always_attached():
    holding = {}
    apply_derived_metrics(holding, PAYLOAD)

    assert holding["look_through"] is True
    assert holding["look_through_n"] == 98
    assert holding["look_through_as_of"] == "2026-08-21"
    assert holding["look_through_coverage"] == {"pe_ratio": 92.5}


def test_stock_payload_follows_the_same_precedence():
    # Both pages must land on the same number for the same fund.
    reported = {"fundamentals": {"peRatio": 30.55, "recommendationMean": None}}
    missing = {"fundamentals": {"peRatio": None, "recommendationMean": None}}
    apply_derived_to_instrument(reported, PAYLOAD)
    apply_derived_to_instrument(missing, PAYLOAD)

    assert reported["fundamentals"]["peRatio"] == 30.55
    assert missing["fundamentals"]["peRatio"] == 31.19
    assert reported["fundamentals"]["recommendationMean"] == 1.69
    assert reported["screener_score"] == missing["screener_score"]


def test_peg_and_fcf_yield_reach_the_stock_page():
    # Both sit on the KPI row beside P/E; Holdings filled them and Stock did not.
    detail = {"fundamentals": {"peRatio": None, "pegRatio": None, "fcfYield": None}}
    apply_derived_to_instrument(detail, PAYLOAD)

    assert detail["fundamentals"]["pegRatio"] == 0.85
    assert detail["fundamentals"]["fcfYield"] == 1.61


def test_an_accumulating_fund_shows_a_zero_yield_not_a_blank():
    # 0.0 is a fact — the fund distributes nothing — and must survive the
    # is-not-None guard that a truthiness check would drop.
    detail = {"fundamentals": {"dividendYield": None}}
    apply_derived_to_instrument(detail, PAYLOAD)

    assert detail["fundamentals"]["dividendYield"] == 0.0
    assert "distribution_yield" in detail["look_through_fields"]


def test_an_unknown_payout_history_stays_blank():
    # None means "we cannot tell", which must not render as 0%.
    detail = {"fundamentals": {"dividendYield": None}}
    apply_derived_to_instrument(detail, {**PAYLOAD, "distribution_yield": None})

    assert detail["fundamentals"]["dividendYield"] is None


def test_a_reported_yield_outranks_the_inferred_one():
    detail = {"fundamentals": {"dividendYield": 3.19}}
    apply_derived_to_instrument(detail, PAYLOAD)

    assert detail["fundamentals"]["dividendYield"] == 3.19


def test_percent_metrics_land_as_yahoo_fractions():
    # Our metrics are percents; Yahoo stores margins and growth as fractions and
    # the Stock page multiplies by 100 to display. A missed factor shows 1837%.
    detail = {"fundamentals": {"profitMargins": None, "returnOnEquity": None, "roic": None}}
    apply_derived_to_instrument(detail, PAYLOAD)

    assert detail["fundamentals"]["profitMargins"] == 0.1837
    assert detail["fundamentals"]["returnOnEquity"] == 0.5381
    assert detail["fundamentals"]["roic"] == 27.9, "roic is a percent on both sides"


def test_conversion_factors_are_only_identity_or_percent_to_fraction():
    # The whole risk of this mapping is a typo in the factor, so pin the set.
    assert {factor for _, factor in FUNDAMENTALS_FIELDS.values()} == {1.0, 0.01}
    for metric in ("profit_margins", "gross_margin", "operating_margin", "revenue_growth", "return_on_equity"):
        assert FUNDAMENTALS_FIELDS[metric][1] == 0.01, metric
    for metric in ("pe_ratio", "peg_ratio", "roic", "free_cashflow_yield"):
        assert FUNDAMENTALS_FIELDS[metric][1] == 1.0, metric


def test_every_writable_field_keeps_the_fund_reported_value():
    # Must hold for every field the overlay can write, not just P/E. dividendYield
    # is written outside FUNDAMENTALS_FIELDS — it comes from payout history.
    writable = {field for field, _ in FUNDAMENTALS_FIELDS.values()} | {"dividendYield"}
    reported = {"fundamentals": {field: 99.0 for field in writable}}
    apply_derived_to_instrument(reported, PAYLOAD)

    assert all(v == 99.0 for v in reported["fundamentals"].values())
    assert reported["look_through_fields"] == ["screener_ratio"], "nothing else should have been filled"


def test_the_two_pages_agree_on_an_identical_fund():
    holding = {"pe_ratio": 30.55, "recommendation_mean": None}
    detail = {"fundamentals": {"peRatio": 30.55, "recommendationMean": None}}
    apply_derived_metrics(holding, PAYLOAD)
    apply_derived_to_instrument(detail, PAYLOAD)

    assert holding["pe_ratio"] == detail["fundamentals"]["peRatio"]
    assert holding["recommendation_mean"] == detail["fundamentals"]["recommendationMean"]
    assert holding["peg_ratio"] == detail["fundamentals"]["pegRatio"]
    assert holding["free_cashflow_yield"] == detail["fundamentals"]["fcfYield"]
    assert holding["screener_score"] == detail["screener_score"]


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
