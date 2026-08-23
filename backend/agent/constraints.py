"""Deterministic constraint layer — final authority over every strategy's output.

Applies the safety caps from config (turnover budget, min holdings, position
and tag-cluster weight caps, min trade size, FX fee) to a list of TradeIntents
and returns constraint-checked TradeOrders. Fully vetoed intents come back as
zero-value orders whose `adjustments` explain the veto, so the UI can show why
a suggestion was blocked instead of silently dropping it.

Sells and buys have separate turnover budgets (each limits.max_daily_turnover
of portfolio value): a de-risking sell should never crowd out a buy — buys
compete with holding cash, not with sells. Each side processes in descending
|score| so its budget goes to the strategy's strongest convictions; ties break
on symbol for determinism. Sells still run first so their proceeds fund buys.

Safety invariants, enforced no matter what a strategy proposes:
  * never sell a symbol that isn't held, never sell more than the position
  * never spend cash that isn't there (start cash + accepted sell proceeds)
  * malformed intents (duplicate symbols, NaN/out-of-range target or score,
    missing price) raise ValueError — a strategy bug fails loud, not quietly
  * post-conditions assert non-negative cash, weights and budgets after the batch
"""

import math

from backend.agent.types import (
    AgentLimits,
    PortfolioState,
    TradeIntent,
    TradeOrder,
    is_etp,
    trade_fee_rate,
)

SELL_ACTIONS = ("exit", "trim")
BUY_ACTIONS = ("buy", "add")

# The one cap tag describing a wrapper rather than a theme.
ETF_TAG = "etf"


def _fee_rate(symbol: str, side: str, state: PortfolioState, limits: AgentLimits) -> float:
    return trade_fee_rate(
        symbol, state.currencies.get(symbol), is_etp(symbol, state.etf_symbols), side, limits
    )


def _cluster_headroom_gbp(
    symbol: str, weights: dict[str, float], state: PortfolioState, limits: AgentLimits
) -> tuple[float, str | None]:
    """Most-binding tag-cluster headroom for adding to `symbol`, and that tag.

    ETFs are exempt from every cap but `etf`, as buyer and as cluster member.
    """
    # Caps limit single-name theme concentration; an index fund is diversified by
    # construction. Counting them also self-blocks the glide path, since XNAS.L is
    # tagged `ai` and XUTC.L `semiconductor` — the clusters they exist to dilute.
    is_etf = symbol in state.etf_symbols
    headroom = float("inf")
    binding_tag = None
    for tag in state.tags.get(symbol, []):
        cap_pct = limits.cluster_caps.get(tag)
        if cap_pct is None:
            continue
        thematic = tag != ETF_TAG
        if thematic and is_etf:
            continue
        cluster_weight = sum(
            w
            for s, w in weights.items()
            if tag in state.tags.get(s, []) and not (thematic and s in state.etf_symbols)
        )
        tag_headroom = (cap_pct / 100.0 - cluster_weight) * state.total_value_gbp
        if tag_headroom < headroom:
            headroom, binding_tag = tag_headroom, tag
    return headroom, binding_tag


def _veto(intent: TradeIntent, weight: float, reason: str, prior: list[str] | None = None) -> TradeOrder:
    return TradeOrder(
        symbol=intent.symbol,
        action=intent.action,
        value_gbp=0.0,
        quantity=0.0,
        weight_before=weight,
        weight_after=weight,
        score=intent.score,
        fee_gbp=0.0,
        rationale=intent.rationale,
        adjustments=tuple(prior or ()) + (reason,),
    )


def apply_constraints(
    intents: list[TradeIntent],
    state: PortfolioState,
    prices_gbp: dict[str, float],
    limits: AgentLimits | None = None,
) -> list[TradeOrder]:
    """Turn strategy intents into safety-capped orders. Deterministic.

    Raises ValueError on malformed input (duplicate symbols, missing
    target_weight, unknown price) — a strategy bug should fail loud, not be
    silently repaired here.
    """
    limits = limits or AgentLimits.from_config()

    symbols = [i.symbol for i in intents]
    if len(symbols) != len(set(symbols)):
        raise ValueError("duplicate symbols in intents")
    for i in intents:
        if i.action in BUY_ACTIONS + ("trim",) and i.target_weight is None:
            raise ValueError(f"{i.symbol}: {i.action} intent requires target_weight")
        if i.target_weight is not None and not (math.isfinite(i.target_weight) and 0.0 <= i.target_weight <= 1.0):
            raise ValueError(f"{i.symbol}: target_weight {i.target_weight} outside [0, 1]")
        if not math.isfinite(i.score):
            raise ValueError(f"{i.symbol}: score must be finite (got {i.score})")
        if i.symbol not in prices_gbp or not math.isfinite(prices_gbp[i.symbol]) or prices_gbp[i.symbol] <= 0:
            raise ValueError(f"{i.symbol}: no positive GBP price")

    total = state.total_value_gbp
    sell_budget_left = limits.max_daily_turnover * total
    buy_budget_left = limits.max_daily_turnover * total
    cash = state.cash_gbp
    weights = dict(state.weights)
    held = {s for s, q in state.quantities.items() if q > 0}

    sells = sorted((i for i in intents if i.action in SELL_ACTIONS), key=lambda i: (-abs(i.score), i.symbol))
    buys = sorted((i for i in intents if i.action in BUY_ACTIONS), key=lambda i: (-abs(i.score), i.symbol))
    orders: list[TradeOrder] = []

    for intent in sells:
        w = weights.get(intent.symbol, 0.0)
        position_value = w * total
        if intent.symbol not in held:
            orders.append(_veto(intent, w, "vetoed: not currently held"))
            continue

        adjustments: list[str] = []
        if intent.action == "exit":
            if len(held) - 1 < limits.min_holdings:
                orders.append(_veto(intent, w, f"vetoed: exit would leave under {limits.min_holdings} holdings"))
                continue
            value = position_value
        else:  # trim
            # Never sell more than the position, whatever the target arithmetic says
            value = min((w - intent.target_weight) * total, position_value)
            if value <= 0:
                orders.append(_veto(intent, w, "vetoed: already at or below trim target"))
                continue

        if value > sell_budget_left:
            adjustments.append(f"clipped by sell budget: wanted £{value:.0f}, allowed £{sell_budget_left:.0f}")
            value = sell_budget_left
        if value <= 0:
            orders.append(_veto(intent, w, "vetoed: daily sell budget already spent", adjustments))
            continue
        # A full exit may fall below min trade size — the position itself is
        # small and blocking it would strand dust positions forever.
        is_full_exit = intent.action == "exit" and value >= position_value
        if value < limits.min_trade_gbp and not is_full_exit:
            orders.append(_veto(intent, w, f"vetoed: £{value:.0f} below min trade size after clipping", adjustments))
            continue

        if is_full_exit:
            quantity = state.quantities[intent.symbol]
        else:
            quantity = min(value / prices_gbp[intent.symbol], state.quantities[intent.symbol])
        fee = value * _fee_rate(intent.symbol, "sell", state, limits)
        new_w = w - value / total
        weights[intent.symbol] = new_w
        if is_full_exit:
            held.discard(intent.symbol)
        cash += value - fee
        sell_budget_left -= value
        orders.append(
            TradeOrder(
                symbol=intent.symbol,
                action=intent.action,
                value_gbp=value,
                quantity=quantity,
                weight_before=w,
                weight_after=new_w,
                score=intent.score,
                fee_gbp=fee,
                rationale=intent.rationale,
                adjustments=tuple(adjustments),
            )
        )

    for intent in buys:
        w = weights.get(intent.symbol, 0.0)
        desired = (intent.target_weight - w) * total
        if desired <= 0:
            orders.append(_veto(intent, w, "vetoed: already at or above target weight"))
            continue

        adjustments: list[str] = []
        value = desired

        position_headroom = (limits.max_position_weight - w) * total
        if position_headroom <= 0:
            orders.append(
                _veto(intent, w, f"vetoed: position at max weight cap ({limits.max_position_weight:.0%})")
            )
            continue
        if value > position_headroom:
            adjustments.append(f"clipped by position weight cap: wanted £{value:.0f}, allowed £{position_headroom:.0f}")
            value = position_headroom

        cluster_headroom, binding_tag = _cluster_headroom_gbp(intent.symbol, weights, state, limits)
        if cluster_headroom <= 0:
            orders.append(_veto(intent, w, f"vetoed: '{binding_tag}' cluster at soft cap"))
            continue
        if value > cluster_headroom:
            adjustments.append(
                f"clipped by '{binding_tag}' cluster cap: wanted £{value:.0f}, allowed £{cluster_headroom:.0f}"
            )
            value = cluster_headroom

        fee_rate = _fee_rate(intent.symbol, "buy", state, limits)
        affordable = cash / (1.0 + fee_rate)
        if value > affordable:
            adjustments.append(f"clipped by available cash: wanted £{value:.0f}, allowed £{affordable:.0f}")
            value = affordable

        if value > buy_budget_left:
            adjustments.append(f"clipped by buy budget: wanted £{value:.0f}, allowed £{buy_budget_left:.0f}")
            value = buy_budget_left
        if value <= 0:
            orders.append(_veto(intent, w, "vetoed: no cash or buy budget left", adjustments))
            continue

        if value < limits.min_trade_gbp:
            orders.append(_veto(intent, w, f"vetoed: £{value:.0f} below min trade size after clipping", adjustments))
            continue

        fee = value * fee_rate
        new_w = w + value / total
        weights[intent.symbol] = new_w
        held.add(intent.symbol)
        cash -= value + fee
        buy_budget_left -= value
        orders.append(
            TradeOrder(
                symbol=intent.symbol,
                action=intent.action,
                value_gbp=value,
                quantity=value / prices_gbp[intent.symbol],
                weight_before=w,
                weight_after=new_w,
                score=intent.score,
                fee_gbp=fee,
                rationale=intent.rationale,
                adjustments=tuple(adjustments),
            )
        )

    # Post-conditions: if any of these trip, the layer itself is buggy — halt
    # rather than emit orders that oversell or overspend.
    assert cash >= -1e-6, f"negative cash after batch: {cash}"
    assert sell_budget_left >= -1e-6 and buy_budget_left >= -1e-6, "turnover budget overspent"
    assert all(v >= -1e-9 for v in weights.values()), "negative position weight after batch"
    return orders
