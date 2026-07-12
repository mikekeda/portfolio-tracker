"""Unit tests for the trade-agent constraint layer.

Pure stdlib — runnable without the app's dependencies:

    python tests/test_agent_constraints.py        # plain asserts
    pytest tests/test_agent_constraints.py        # also pytest-compatible
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.agent.constraints import apply_constraints
from backend.agent.types import AgentLimits, PortfolioState, TradeIntent

LIMITS = AgentLimits(
    max_daily_turnover=0.05,
    min_holdings=5,
    max_position_weight=0.16,
    min_trade_gbp=150.0,
    fx_fee=0.0015,
    cluster_caps={"ai": 45, "space": 15},
)


def make_state(**overrides) -> PortfolioState:
    """Six-holding £100k portfolio: NVDA 15%, GOOGL 10%, ASML 10%, RKLB 5%, VUAG 8%, MU 2%; £5k cash."""
    defaults = dict(
        date=date(2026, 7, 10),
        total_value_gbp=100_000.0,
        cash_gbp=5_000.0,
        weights={"NVDA": 0.15, "GOOGL": 0.10, "ASML": 0.10, "RKLB": 0.05, "VUAG.L": 0.08, "MU": 0.02},
        quantities={"NVDA": 100, "GOOGL": 50, "ASML": 20, "RKLB": 200, "VUAG.L": 120, "MU": 30},
        currencies={"NVDA": "USD", "GOOGL": "USD", "ASML": "USD", "RKLB": "USD", "VUAG.L": "GBP", "MU": "USD"},
        tags={"NVDA": ["ai"], "GOOGL": ["ai"], "ASML": ["ai"], "RKLB": ["space"], "VUAG.L": ["etf"], "MU": ["ai"]},
    )
    defaults.update(overrides)
    return PortfolioState(**defaults)


PRICES = {"NVDA": 120.0, "GOOGL": 150.0, "ASML": 700.0, "RKLB": 25.0, "VUAG.L": 65.0, "MU": 90.0, "AMD": 110.0}


def by_symbol(orders):
    return {o.symbol: o for o in orders}


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_min_holdings_vetoes_exit():
    state = make_state()
    # 6 holdings; two exits would leave 4 < 5 — the lower-score one must be vetoed.
    intents = [
        TradeIntent("MU", "exit", None, score=-3.0),
        TradeIntent("RKLB", "exit", None, score=-2.0),
    ]
    orders = by_symbol(apply_constraints(intents, state, PRICES, LIMITS))
    assert orders["MU"].executable, orders["MU"]
    assert not orders["RKLB"].executable
    assert "under 5 holdings" in orders["RKLB"].adjustments[0]


def test_full_exit_uses_exact_quantity_and_allows_dust():
    state = make_state()
    # MU position = £2,000; a full exit sells exactly the held quantity.
    orders = by_symbol(apply_constraints([TradeIntent("MU", "exit", None, -1.0)], state, PRICES, LIMITS))
    assert orders["MU"].quantity == 30
    assert orders["MU"].value_gbp == 2_000.0
    # A £100 dust position exits despite being under min_trade_gbp.
    state = make_state(
        weights={**make_state().weights, "MU": 0.001},
        total_value_gbp=100_000.0,
    )
    orders = by_symbol(apply_constraints([TradeIntent("MU", "exit", None, -1.0)], state, PRICES, LIMITS))
    assert orders["MU"].executable and orders["MU"].value_gbp == 100.0


def test_turnover_budget_goes_to_highest_conviction():
    state = make_state()
    # Budget = £5,000. Trim NVDA to 12% (£3,000, score 5) + trim GOOGL to 7% (£3,000, score 2):
    # NVDA passes whole, GOOGL is clipped to the remaining £2,000.
    intents = [
        TradeIntent("GOOGL", "trim", 0.07, score=2.0),
        TradeIntent("NVDA", "trim", 0.12, score=5.0),
    ]
    orders = by_symbol(apply_constraints(intents, state, PRICES, LIMITS))
    assert orders["NVDA"].value_gbp == 3_000.0 and not orders["NVDA"].adjustments
    assert orders["GOOGL"].value_gbp == 2_000.0
    assert "turnover budget" in orders["GOOGL"].adjustments[0]


def test_position_weight_cap_clips_add():
    state = make_state(cash_gbp=50_000.0)
    limits = AgentLimits(**{**LIMITS.__dict__, "max_daily_turnover": 0.50, "cluster_caps": {}})
    # NVDA at 15%, cap 16% — an add to 20% is clipped to £1,000 (1% of portfolio).
    orders = by_symbol(apply_constraints([TradeIntent("NVDA", "add", 0.20, 4.0)], state, PRICES, limits))
    assert approx(orders["NVDA"].value_gbp, 1_000.0), orders["NVDA"]
    assert "position weight cap" in orders["NVDA"].adjustments[0]


def test_cluster_cap_binds_across_holdings():
    # ai cluster = NVDA 15 + GOOGL 10 + ASML 10 + MU 2 = 37%; cap 45% → £8k headroom.
    state = make_state(cash_gbp=50_000.0)
    limits = AgentLimits(**{**LIMITS.__dict__, "max_daily_turnover": 0.50})
    state.tags["AMD"] = ["ai"]
    state.currencies["AMD"] = "USD"
    orders = by_symbol(apply_constraints([TradeIntent("AMD", "buy", 0.12, 4.0)], state, PRICES, limits))
    assert approx(orders["AMD"].value_gbp, 8_000.0), orders["AMD"]
    assert "'ai' cluster cap" in orders["AMD"].adjustments[0]


def test_sell_proceeds_fund_buys():
    # Cash £0: buy alone is unaffordable, but paired with an exit it proceeds.
    state = make_state(cash_gbp=0.0)
    state.tags["AMD"] = []
    state.currencies["AMD"] = "USD"
    solo = by_symbol(apply_constraints([TradeIntent("AMD", "buy", 0.02, 3.0)], state, PRICES, LIMITS))
    assert not solo["AMD"].executable
    paired = by_symbol(
        apply_constraints(
            [TradeIntent("AMD", "buy", 0.02, 3.0), TradeIntent("MU", "exit", None, -4.0)],
            state,
            PRICES,
            LIMITS,
        )
    )
    assert paired["MU"].executable
    assert paired["AMD"].executable
    # £2,000 sale minus the USD FX fee funds the buy net of its own fee.
    assert paired["AMD"].value_gbp < 2_000.0


def test_fx_fee_only_on_non_sterling():
    state = make_state()
    intents = [
        TradeIntent("VUAG.L", "trim", 0.06, score=1.0),
        TradeIntent("NVDA", "trim", 0.14, score=2.0),
    ]
    orders = by_symbol(apply_constraints(intents, state, PRICES, LIMITS))
    assert orders["VUAG.L"].fee_gbp == 0.0
    assert abs(orders["NVDA"].fee_gbp - 1_000.0 * 0.0015) < 1e-9


def test_malformed_intents_fail_loud():
    state = make_state()
    for bad in (
        [TradeIntent("NVDA", "trim", None, 1.0)],  # trim without target
        [TradeIntent("NVDA", "exit", None, 1.0), TradeIntent("NVDA", "trim", 0.1, 1.0)],  # dupe
        [TradeIntent("ZZZZ", "buy", 0.02, 1.0)],  # no price
    ):
        try:
            apply_constraints(bad, state, PRICES, LIMITS)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")


def test_deterministic():
    state = make_state()
    intents = [
        TradeIntent("GOOGL", "trim", 0.07, score=2.0),
        TradeIntent("NVDA", "trim", 0.12, score=5.0),
        TradeIntent("RKLB", "add", 0.06, score=2.0),
    ]
    first = apply_constraints(intents, state, PRICES, LIMITS)
    second = apply_constraints(list(reversed(intents)), make_state(), PRICES, LIMITS)
    assert first == second


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
