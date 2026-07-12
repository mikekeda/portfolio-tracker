"""Deterministic constraint layer — final authority over every strategy's output.

Applies the safety caps from config (turnover budget, min holdings, position
and tag-cluster weight caps, min trade size, FX fee) to a list of TradeIntents
and returns constraint-checked TradeOrders. Fully vetoed intents come back as
zero-value orders whose `adjustments` explain the veto, so the UI can show why
a suggestion was blocked instead of silently dropping it.

Processing order: sells first (they free cash and turnover for nothing), then
buys — each group in descending |score| so the turnover budget goes to the
strategy's strongest convictions. Ties break on symbol for determinism.
"""

from backend.agent.types import STERLING, AgentLimits, PortfolioState, TradeIntent, TradeOrder

SELL_ACTIONS = ("exit", "trim")
BUY_ACTIONS = ("buy", "add")


def _fee(value: float, symbol: str, state: PortfolioState, limits: AgentLimits) -> float:
    return 0.0 if state.currencies.get(symbol) in STERLING else value * limits.fx_fee


def _cluster_headroom_gbp(
    symbol: str, weights: dict[str, float], state: PortfolioState, limits: AgentLimits
) -> tuple[float, str | None]:
    """Most-binding tag-cluster headroom for adding to `symbol`, and that tag."""
    headroom = float("inf")
    binding_tag = None
    for tag in state.tags.get(symbol, []):
        cap_pct = limits.cluster_caps.get(tag)
        if cap_pct is None:
            continue
        cluster_weight = sum(w for s, w in weights.items() if tag in state.tags.get(s, []))
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
        if i.symbol not in prices_gbp or prices_gbp[i.symbol] <= 0:
            raise ValueError(f"{i.symbol}: no positive GBP price")

    total = state.total_value_gbp
    turnover_left = limits.max_daily_turnover * total
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
            value = (w - intent.target_weight) * total
            if value <= 0:
                orders.append(_veto(intent, w, "vetoed: already at or below trim target"))
                continue

        if value > turnover_left:
            adjustments.append(f"clipped by turnover budget: wanted £{value:.0f}, allowed £{turnover_left:.0f}")
            value = turnover_left
        if value <= 0:
            orders.append(_veto(intent, w, "vetoed: daily turnover budget already spent", adjustments))
            continue
        # A full exit may fall below min trade size — the position itself is
        # small and blocking it would strand dust positions forever.
        is_full_exit = intent.action == "exit" and value >= position_value
        if value < limits.min_trade_gbp and not is_full_exit:
            orders.append(_veto(intent, w, f"vetoed: £{value:.0f} below min trade size after clipping", adjustments))
            continue

        quantity = state.quantities[intent.symbol] if is_full_exit else value / prices_gbp[intent.symbol]
        fee = _fee(value, intent.symbol, state, limits)
        new_w = w - value / total
        weights[intent.symbol] = new_w
        if is_full_exit:
            held.discard(intent.symbol)
        cash += value - fee
        turnover_left -= value
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

        fee_rate = 0.0 if state.currencies.get(intent.symbol) in STERLING else limits.fx_fee
        affordable = cash / (1.0 + fee_rate)
        if value > affordable:
            adjustments.append(f"clipped by available cash: wanted £{value:.0f}, allowed £{affordable:.0f}")
            value = affordable

        if value > turnover_left:
            adjustments.append(f"clipped by turnover budget: wanted £{value:.0f}, allowed £{turnover_left:.0f}")
            value = turnover_left
        if value <= 0:
            orders.append(_veto(intent, w, "vetoed: no cash or turnover budget left", adjustments))
            continue

        if value < limits.min_trade_gbp:
            orders.append(_veto(intent, w, f"vetoed: £{value:.0f} below min trade size after clipping", adjustments))
            continue

        fee = value * fee_rate
        new_w = w + value / total
        weights[intent.symbol] = new_w
        held.add(intent.symbol)
        cash -= value + fee
        turnover_left -= value
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

    return orders
