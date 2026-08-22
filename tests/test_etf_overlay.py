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
from backend.views._etf_overlay import apply_derived_metrics, apply_derived_to_instrument

# XNAS.L: Yahoo reports a real trailingPE of 30.55 for the fund; our constituent
# aggregate says 31.2. UKDV.L is the case where the two visibly disagree.
PAYLOAD = {
    "metrics": {
        "pe_ratio": 31.19,
        "roic": 27.9,
        "recommendation_mean": 1.6937,
        "form13f_score": 0.8,
        "screener_ratio": 0.427,
    },
    "coverage": {"pe_ratio": 92.5},
    "n_resolved": 98,
    "as_of": "2026-08-21",
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


def test_the_two_pages_agree_on_an_identical_fund():
    holding = {"pe_ratio": 30.55, "recommendation_mean": None}
    detail = {"fundamentals": {"peRatio": 30.55, "recommendationMean": None}}
    apply_derived_metrics(holding, PAYLOAD)
    apply_derived_to_instrument(detail, PAYLOAD)

    assert holding["pe_ratio"] == detail["fundamentals"]["peRatio"]
    assert holding["recommendation_mean"] == detail["fundamentals"]["recommendationMean"]
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
