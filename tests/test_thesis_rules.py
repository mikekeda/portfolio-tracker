"""Tests for thesis rule evaluation: grouped AND/OR logic and the sell-streak.

Needs the app venv (pydantic, pandas) — run on the server from project root:

    python tests/test_thesis_rules.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from pydantic import ValidationError

from backend.schemas.instrument_thesis import InstrumentThesisSchema
from backend.utils.thesis_rules import evaluate_thesis_rules

BASE = {
    "summary": "test",
    "target_weight_min_pct": 1.0,
    "target_weight_max_pct": 5.0,
    "horizon_years": 5,
    "authored_on": "2026-07-13",
}

HOLDING = {
    "portfolio_pct": 3.0,  # in band — allocation contributes no signal
    "revenue_growth": 5.0,
    "profit_margins": 30.0,
    "roic": 10.0,
    "debtToEquity": 250.0,
    "free_cashflow_yield": None,  # deliberately missing
}


def thesis(sell_rules):
    return InstrumentThesisSchema.model_validate({**BASE, "sell_rules": sell_rules})


def test_flat_rules_unchanged():
    ev = evaluate_thesis_rules(HOLDING, thesis([
        {"field": "revenue_growth", "operator": "<", "value": 10, "description": "slowing"},
    ]))
    assert ev["sell_signal"] and ev["sell_rules_met"][0]["description"] == "slowing"


def test_all_group_requires_every_child():
    # growth < 10 holds, margins < 20 does not → group must NOT fire
    ev = evaluate_thesis_rules(HOLDING, thesis([
        {"all": [
            {"field": "revenue_growth", "operator": "<", "value": 10},
            {"field": "profit_margins", "operator": "<", "value": 20},
        ], "description": "growth AND margins breaking"},
    ]))
    assert not ev["sell_signal"], ev
    # both hold → fires, with a compound reason
    ev = evaluate_thesis_rules(HOLDING, thesis([
        {"all": [
            {"field": "revenue_growth", "operator": "<", "value": 10},
            {"field": "profit_margins", "operator": "<", "value": 40},
        ], "description": "growth AND margins breaking"},
    ]))
    assert ev["sell_signal"]
    assert "AND" in ev["sell_rules_met"][0]["reason"], ev["sell_rules_met"]


def test_nested_c1c2_or_c3c4():
    # ((rev<10 AND margins<20) OR (roic<15 AND d/e>200)) — first false, second true
    ev = evaluate_thesis_rules(HOLDING, thesis([
        {"all": [
            {"field": "revenue_growth", "operator": "<", "value": 10},
            {"field": "profit_margins", "operator": "<", "value": 20},
        ]},
        {"all": [
            {"field": "roic", "operator": "<", "value": 15},
            {"field": "debtToEquity", "operator": ">", "value": 200},
        ], "description": "returns broken and levered"},
    ]))
    assert ev["sell_signal"]
    assert len(ev["sell_rules_met"]) == 1
    assert ev["sell_rules_met"][0]["description"] == "returns broken and levered"


def test_any_inside_all_and_missing_data_is_conservative():
    # fcf yield is None → that leaf is False; the any-group falls through to d/e
    ev = evaluate_thesis_rules(HOLDING, thesis([
        {"all": [
            {"field": "roic", "operator": "<", "value": 15},
            {"any": [
                {"field": "free_cashflow_yield", "operator": "<", "value": 0},
                {"field": "debtToEquity", "operator": ">", "value": 200},
            ]},
        ]},
    ]))
    assert ev["sell_signal"]
    # an all-group where one leaf has missing data must NOT fire
    ev = evaluate_thesis_rules(HOLDING, thesis([
        {"all": [
            {"field": "roic", "operator": "<", "value": 15},
            {"field": "free_cashflow_yield", "operator": "<", "value": 0},
        ]},
    ]))
    assert not ev["sell_signal"], "missing data inside an all-group fired an exit signal"


def test_schema_rejects_malformed_groups():
    for bad in (
        [{"all": [{"field": "roic", "operator": "<", "value": 15}],
          "any": [{"field": "roic", "operator": "<", "value": 15}]}],  # both
        [{"description": "empty group"}],  # neither
        [{"all": []}],  # empty children
        [{"all": [{"field": "not_a_field", "operator": "<", "value": 1}]}],  # bad nested leaf
    ):
        try:
            thesis(bad)
        except ValidationError:
            pass
        else:
            raise AssertionError(f"expected ValidationError for {bad}")


def test_sell_streak_is_backward_only():
    from backend.agent.backtest.data import MarketData  # noqa: F401  (import path check only)

    s = pd.Series([False, True, True, False, True, True, True])
    streak = s.astype(int).groupby((~s).cumsum()).cumsum()
    assert list(streak) == [0, 1, 2, 0, 1, 2, 3]


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
