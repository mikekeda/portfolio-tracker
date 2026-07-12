"""
Evaluate thesis buy_rules / sell_rules against holdings data.

Buy/sell signals use OR semantics within each rule list. Allocation band
(target_weight_min/max_pct vs portfolio_pct) also contributes: below min → buy,
above max → sell. When both sides fire, conflict is true and both signals show.
"""

from typing import Any

from pydantic import ValidationError

from backend.screener_config import FieldRef, ScreenerCriteria, get_screener_config
from backend.schemas.instrument_thesis import InstrumentThesisSchema, ThesisRuleSchema
from config import logger
from models import InstrumentThesisTypedDict

AllocationStatus = str  # below_min | in_band | above_max | unknown


def _rule_to_criteria(rule: ThesisRuleSchema) -> ScreenerCriteria:
    value: Any = rule.value
    if isinstance(value, dict) and "fieldRef" in value:
        value = FieldRef(value["fieldRef"])
    return ScreenerCriteria(
        field=rule.field,
        operator=rule.operator,
        value=value,
        description=rule.description,
    )


def _rules_met(holding: dict, rules: list[ThesisRuleSchema]) -> list[dict[str, Any]]:
    config = get_screener_config()
    met: list[dict[str, Any]] = []
    for rule in rules:
        passed, reason = config.eval_criterion(holding, _rule_to_criteria(rule))
        if passed:
            met.append({
                "description": rule.description or reason,
                "reason": reason,
            })
    return met


def _allocation_status(holding: dict, thesis: InstrumentThesisSchema) -> AllocationStatus:
    pct = holding["portfolio_pct"]
    if pct < thesis.target_weight_min_pct:
        return "below_min"
    if pct > thesis.target_weight_max_pct:
        return "above_max"
    return "in_band"


def _allocation_reason(
    holding: dict,
    thesis: InstrumentThesisSchema,
    status: AllocationStatus,
) -> str | None:
    if status == "in_band":
        return None
    pct = holding["portfolio_pct"]
    if status == "below_min":
        return f"Weight {pct:.1f}% below target min {thesis.target_weight_min_pct}%"
    return f"Weight {pct:.1f}% above target max {thesis.target_weight_max_pct}%"


def _buy_reasons(ev: dict[str, Any]) -> list[str]:
    reasons = [r["description"] for r in ev["buy_rules_met"]]
    if ev["allocation_status"] == "below_min":
        reasons.append(ev["allocation_reason"])
    return reasons


def _sell_reasons(ev: dict[str, Any]) -> list[str]:
    reasons = [r["description"] for r in ev["sell_rules_met"]]
    if ev["allocation_status"] == "above_max":
        reasons.append(ev["allocation_reason"])
    return reasons


def evaluate_thesis_rules(
    holding: dict,
    thesis: InstrumentThesisSchema,
) -> dict[str, Any]:
    """Evaluate thesis rules and allocation band for one holding dict."""
    eval_holding = {
        **holding,
        "target_weight_min_pct": thesis.target_weight_min_pct,
        "target_weight_max_pct": thesis.target_weight_max_pct,
    }
    buy_rules_met = _rules_met(eval_holding, thesis.buy_rules)
    sell_rules_met = _rules_met(eval_holding, thesis.sell_rules)
    allocation_status = _allocation_status(eval_holding, thesis)
    allocation_reason = _allocation_reason(eval_holding, thesis, allocation_status)

    buy_signal = bool(buy_rules_met) or allocation_status == "below_min"
    sell_signal = bool(sell_rules_met) or allocation_status == "above_max"
    return {
        "buy_signal": buy_signal,
        "sell_signal": sell_signal,
        "conflict": buy_signal and sell_signal,
        "buy_rules_met": buy_rules_met,
        "sell_rules_met": sell_rules_met,
        "allocation_status": allocation_status,
        "allocation_reason": allocation_reason,
        # Band bounds so consumers (trade agent) can trim to the band instead
        # of guessing; persisted into FeaturesDaily.thesis_rule_eval.
        "target_weight_min_pct": thesis.target_weight_min_pct,
        "target_weight_max_pct": thesis.target_weight_max_pct,
    }


def attach_thesis_rule_eval(
    portfolio_data: list[dict],
    thesis_by_symbol: dict[str, InstrumentThesisTypedDict],
) -> dict[str, list[dict[str, Any]]]:
    """Attach thesis_rule_eval to each holding and return rule_recommendations."""

    buy: list[dict[str, Any]] = []
    sell: list[dict[str, Any]] = []

    for h in portfolio_data:
        # Set default thesis_rule_eval value for each holding
        h["thesis_rule_eval"] = {
            "buy_signal": False,
            "sell_signal": False,
            "conflict": False,
            "buy_rules_met": [],
            "sell_rules_met": [],
            "allocation_status": "unknown",
            "allocation_reason": None,
            "target_weight_min_pct": None,
            "target_weight_max_pct": None,
        }

        if raw := thesis_by_symbol.get(h["yahoo_symbol"]):
            try:
                thesis = InstrumentThesisSchema.model_validate(raw)
            except ValidationError as e:
                logger.info("Invalid thesis for %s: %s", h["yahoo_symbol"], e.errors())
                continue

            ev = evaluate_thesis_rules(h, thesis)
            h["thesis_rule_eval"] = ev

            if ev["buy_signal"]:
                buy.append({
                    "symbol": h["yahoo_symbol"],
                    "name": h["name"],
                    "conflict": ev["conflict"],
                    "reasons": _buy_reasons(ev),
                })
            if ev["sell_signal"]:
                sell.append({
                    "symbol": h["yahoo_symbol"],
                    "name": h["name"],
                    "conflict": ev["conflict"],
                    "reasons": _sell_reasons(ev),
                })

    return {"buy": buy, "sell": sell}
