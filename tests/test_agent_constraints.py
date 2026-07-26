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
    stamp_duty=0.005,
    french_ftt=0.004,
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
    assert "sell budget" in orders["GOOGL"].adjustments[0]


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


def test_etfs_exempt_from_thematic_cluster_caps():
    # ai = 37% of a 45% cap, plus a 10% ai-tagged ETF: counting it would put the
    # cluster at 47% and veto every ai buy, index core included.
    state = make_state(cash_gbp=50_000.0)
    limits = AgentLimits(**{**LIMITS.__dict__, "max_daily_turnover": 0.50})
    state.weights["XNAS.L"] = 0.10
    state.quantities["XNAS.L"] = 100
    state.currencies["XNAS.L"] = "GBP"
    state.tags["XNAS.L"] = ["ai", "growth", "etf"]
    state.etf_symbols = frozenset({"VUAG.L", "XNAS.L"})
    prices = {**PRICES, "XNAS.L": 100.0}

    etf_order = by_symbol(
        apply_constraints([TradeIntent("XNAS.L", "add", 0.16, 4.0)], state, prices, limits)
    )["XNAS.L"]
    assert etf_order.executable, etf_order
    assert not any("cluster cap" in a for a in etf_order.adjustments), etf_order.adjustments

    # A direct ai name still gets exactly the £8k it had before the ETF existed.
    state.tags["AMD"] = ["ai"]
    state.currencies["AMD"] = "USD"
    direct = by_symbol(apply_constraints([TradeIntent("AMD", "buy", 0.12, 4.0)], state, prices, limits))["AMD"]
    assert approx(direct.value_gbp, 8_000.0), direct
    assert "'ai' cluster cap" in direct.adjustments[0]


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
    state = make_state(etf_symbols=frozenset({"VUAG.L"}))
    intents = [
        TradeIntent("VUAG.L", "trim", 0.06, score=1.0),
        TradeIntent("NVDA", "trim", 0.14, score=2.0),
    ]
    orders = by_symbol(apply_constraints(intents, state, PRICES, LIMITS))
    assert orders["VUAG.L"].fee_gbp == 0.0
    assert abs(orders["NVDA"].fee_gbp - 1_000.0 * 0.0015) < 1e-9


def test_buys_have_own_budget_not_crowded_out_by_sells():
    # Sells exhaust their entire 5% budget; a buy must still go through on its own budget.
    state = make_state(cash_gbp=5_000.0)
    state.tags["AMD"] = []
    state.currencies["AMD"] = "USD"
    intents = [
        TradeIntent("NVDA", "trim", 0.10, score=9.0),  # £5,000 — consumes full sell budget
        TradeIntent("AMD", "buy", 0.03, score=1.0),  # £3,000 — must not be blocked
    ]
    orders = by_symbol(apply_constraints(intents, state, PRICES, LIMITS))
    assert approx(orders["NVDA"].value_gbp, 5_000.0), orders["NVDA"]
    assert orders["AMD"].executable, orders["AMD"]
    assert approx(orders["AMD"].value_gbp, 3_000.0, tol=25), orders["AMD"]


def test_stamp_duty_on_uk_share_buys_not_etfs_or_sells():
    state = make_state(cash_gbp=20_000.0, etf_symbols=frozenset({"VUAG.L"}))
    prices = {**PRICES, "RR.L": 5.0}
    state.tags["RR.L"] = []
    state.currencies["RR.L"] = "GBX"
    # UK share buy: 0.5% stamp duty, no FX fee.
    orders = by_symbol(apply_constraints([TradeIntent("RR.L", "buy", 0.02, 1.0)], state, prices, LIMITS))
    assert approx(orders["RR.L"].fee_gbp, 2_000.0 * 0.005), orders["RR.L"]
    # ETF buy in GBP: no stamp duty, no FX fee.
    orders = by_symbol(apply_constraints([TradeIntent("VUAG.L", "add", 0.10, 1.0)], state, prices, LIMITS))
    assert orders["VUAG.L"].fee_gbp == 0.0, orders["VUAG.L"]
    # UK share sell: no stamp duty either.
    state2 = make_state(
        weights={**make_state().weights, "RR.L": 0.04},
        quantities={**make_state().quantities, "RR.L": 800},
        etf_symbols=frozenset({"VUAG.L"}),
    )
    state2.currencies["RR.L"] = "GBX"
    orders = by_symbol(apply_constraints([TradeIntent("RR.L", "trim", 0.02, -1.0)], state2, prices, LIMITS))
    assert orders["RR.L"].fee_gbp == 0.0, orders["RR.L"]


def test_french_ftt_plus_fx_on_pa_buys():
    state = make_state(cash_gbp=20_000.0)
    prices = {**PRICES, "AM.PA": 60.0}
    state.tags["AM.PA"] = []
    state.currencies["AM.PA"] = "EUR"
    orders = by_symbol(apply_constraints([TradeIntent("AM.PA", "buy", 0.02, 1.0)], state, prices, LIMITS))
    assert approx(orders["AM.PA"].fee_gbp, 2_000.0 * (0.0015 + 0.004), tol=0.5), orders["AM.PA"]


def test_malformed_intents_fail_loud():
    state = make_state()
    nan = float("nan")
    for bad in (
        [TradeIntent("NVDA", "trim", None, 1.0)],  # trim without target
        [TradeIntent("NVDA", "exit", None, 1.0), TradeIntent("NVDA", "trim", 0.1, 1.0)],  # dupe
        [TradeIntent("ZZZZ", "buy", 0.02, 1.0)],  # no price
        [TradeIntent("NVDA", "trim", -0.05, 1.0)],  # negative target would oversell
        [TradeIntent("NVDA", "add", 1.5, 1.0)],  # target above 100%
        [TradeIntent("NVDA", "add", nan, 1.0)],  # NaN target
        [TradeIntent("NVDA", "exit", None, nan)],  # NaN score breaks deterministic ranking
    ):
        try:
            apply_constraints(bad, state, PRICES, LIMITS)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")


def test_cannot_sell_unheld_symbol():
    state = make_state()
    state.currencies["AMD"] = "USD"
    state.tags["AMD"] = []
    orders = by_symbol(apply_constraints([TradeIntent("AMD", "exit", None, -2.0)], state, PRICES, LIMITS))
    assert not orders["AMD"].executable
    assert "not currently held" in orders["AMD"].adjustments[0]


def test_sell_never_exceeds_position():
    # Trim to 0% must sell exactly the position, never more (short-sell guard);
    # relaxed budget so the position cap is the only binding constraint.
    limits = AgentLimits(**{**LIMITS.__dict__, "max_daily_turnover": 1.0})
    state = make_state()
    orders = by_symbol(apply_constraints([TradeIntent("MU", "trim", 0.0, -1.0)], state, PRICES, limits))
    assert approx(orders["MU"].value_gbp, 2_000.0), orders["MU"]  # full £2,000 position, not more
    assert orders["MU"].quantity <= state.quantities["MU"] + 1e-9


def test_buys_capped_by_cash_including_sell_proceeds():
    # £0 cash: buy is funded exclusively by the exit's proceeds and cannot exceed them.
    limits = AgentLimits(**{**LIMITS.__dict__, "max_daily_turnover": 1.0, "cluster_caps": {}})
    state = make_state(cash_gbp=0.0)
    state.currencies["AMD"] = "USD"
    state.tags["AMD"] = []
    intents = [
        TradeIntent("MU", "exit", None, -3.0),  # £2,000 proceeds (minus FX fee)
        TradeIntent("AMD", "buy", 0.10, 2.0),  # wants £10,000
    ]
    orders = by_symbol(apply_constraints(intents, state, PRICES, limits))
    proceeds = 2_000.0 * (1 - 0.0015)
    max_buy = proceeds / 1.0015
    assert approx(orders["AMD"].value_gbp, max_buy, tol=1.0), orders["AMD"]
    assert "clipped by available cash" in orders["AMD"].adjustments[0]


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
