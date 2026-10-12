"""Thesis band max caps adds by starvation and never triggers a trim (STRATEGY.md R2)."""

from datetime import date

import numpy as np
import pandas as pd

from backend.agent.rules_strategy import RulesStrategy
from backend.agent.types import AgentLimits, PortfolioState

LIMITS = AgentLimits(
    max_daily_turnover=0.05,
    min_holdings=5,
    max_position_weight=0.16,
    min_trade_gbp=150.0,
    fx_fee=0.0015,
)

WEIGHTS = {"CAPPED": 0.14, "NEAR": 0.10, "C": 0.22, "D": 0.22, "E": 0.22}

# CAPPED and NEAR are the top two by momentum, so both clear ADD_PCTL and sit
# well under their HRP targets; only the thesis band can hold them back.
FEATURES = pd.DataFrame(
    {
        "mom_12_1": [5.0, 4.0, 1.0, 0.0, -1.0],
        "dist_sma200": [0.1] * 5,
        "hrp_weight": [0.25, 0.20, 0.20, 0.20, 0.15],
        "risk_contribution": [0.10, 0.10, 0.20, 0.20, 0.15],
        "thesis_target_max": [12.0, 11.0, np.nan, np.nan, np.nan],
        "thesis_sell_fired": [False] * 5,
        "thesis_sell_streak": [0] * 5,
    },
    index=list(WEIGHTS),
)


def _intents(features):
    state = PortfolioState(
        date=date(2026, 10, 12),
        total_value_gbp=100_000.0,
        cash_gbp=10_000.0,
        weights=WEIGHTS,
        quantities={s: 1.0 for s in WEIGHTS},
        currencies={s: "USD" for s in WEIGHTS},
    )
    intents = RulesStrategy(LIMITS).propose(date(2026, 10, 12), features, state)
    return {i.symbol: (i.action, i.target_weight) for i in intents}


def test_position_above_band_max_is_neither_trimmed_nor_added_to():
    assert "CAPPED" not in _intents(FEATURES)


def test_add_stops_at_band_max():
    action, target = _intents(FEATURES)["NEAR"]
    assert action == "add" and target == 0.11


def test_add_without_thesis_is_bounded_by_the_step_only():
    action, target = _intents(FEATURES.assign(thesis_target_max=np.nan))["NEAR"]
    assert action == "add" and abs(target - 0.12) < 1e-9


# SMALL's band is below the flat opening weight, BANNED's band max of 0 means
# "don't own" and must not take one of the MAX_NEW_BUYS slots from FREE.
NEW = pd.DataFrame(
    {"dist_sma200": [0.1] * 3, "thesis_target_max": [1.5, 0.0, np.nan]},
    index=["SMALL", "BANNED", "FREE"],
)
NEW_COMP = pd.Series({"SMALL": 2.0, "BANNED": 1.9, "FREE": 1.8})
NEW_PCTL = pd.Series({"SMALL": 0.99, "BANNED": 0.98, "FREE": 0.97})


def test_new_buy_opens_at_most_at_band_max():
    buys = RulesStrategy(LIMITS)._new_buys(NEW, NEW_COMP, NEW_PCTL, set())
    assert {i.symbol: i.target_weight for i in buys} == {"SMALL": 0.015, "FREE": RulesStrategy.NEW_POSITION_WEIGHT}


def test_deploy_opens_at_most_at_band_max():
    state = PortfolioState(
        date=date(2026, 10, 12),
        total_value_gbp=100_000.0,
        cash_gbp=90_000.0,
        weights={},
        quantities={},
        currencies={s: "USD" for s in NEW.index},
    )
    intents = RulesStrategy(LIMITS)._deploy(NEW, NEW_COMP, state)
    assert {i.symbol: i.target_weight for i in intents} == {"SMALL": 0.015, "FREE": LIMITS.max_position_weight}
