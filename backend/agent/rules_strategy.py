"""RulesStrategy — transparent cross-sectional composite scoring (agent v1).

Each feature is z-scored across the day's universe and combined with fixed
weights into one conviction score per symbol. Tier-A weights (price-derived)
are always active; Tier-B weights (fundamentals from FeaturesDaily) activate
per symbol as data exists, and the weight sum renormalizes over available
features — so a 2018 backtest scores honestly on technicals alone instead of
pretending fundamentals were known.

Intent rules (all thresholds are class attributes, tunable via backtest):
  EXIT  held, bottom decile composite and below 200d SMA — or thesis sell rule
  TRIM  risk contribution far above HRP weight and below-median composite
  ADD   top quintile composite, underweight vs HRP target
  BUY   non-held, top decile composite, uptrend, quality gate when Tier B known
  DEPLOY when cash > 30% of the book (backtest start / large deposit): buy the
         top-ranked quality names toward an equal-weight book
"""

from datetime import date

import numpy as np
import pandas as pd

from backend.agent.types import AgentLimits, PortfolioState, TradeIntent

TIER_A_WEIGHTS = {
    "mom_12_1": 2.0,
    "mom_6m": 1.0,
    "rs_6m_vs_spy": 1.5,
    "dist_sma200": 1.0,
    "dist_52w_high": 0.5,
    "rsi_14": -0.5,
    "vol_90d": -1.0,
}
TIER_B_WEIGHTS = {
    "screener_score": 2.0,
    "roic": 1.0,
    "rule_of_40": 1.0,
    "fcf_yield": 0.5,
    "dcf_implied_growth": -0.5,
}
Z_CLIP = 3.0


def composite_score(features: pd.DataFrame) -> pd.Series:
    """Weighted sum of clipped cross-sectional z-scores, NaN-renormalized."""
    weights = {**TIER_A_WEIGHTS, **TIER_B_WEIGHTS}
    cols = [c for c in weights if c in features.columns]
    z = pd.DataFrame(index=features.index)
    for c in cols:
        x = pd.to_numeric(features[c], errors="coerce")
        std = x.std()
        z[c] = ((x - x.mean()) / std).clip(-Z_CLIP, Z_CLIP) if std and std > 0 else np.nan

    w = pd.Series({c: weights[c] for c in z.columns})
    weighted = z.mul(w, axis=1).sum(axis=1)
    norm = z.notna().mul(w.abs(), axis=1).sum(axis=1)
    return weighted / norm.replace(0.0, np.nan)


class RulesStrategy:
    name = "rules"

    EXIT_PCTL = 0.10
    ADD_PCTL = 0.80
    BUY_PCTL = 0.90
    MAX_NEW_BUYS = 2
    NEW_POSITION_WEIGHT = 0.02
    STEP_WEIGHT = 0.02  # max add/trim move toward the HRP target per rebalance
    RC_EXCESS_TRIM = 0.05  # trim when risk contribution exceeds HRP weight by this
    DEPLOY_CASH_FRAC = 0.30
    DEPLOY_POSITIONS = 15
    MIN_QUALITY_ROIC = 15.0
    MIN_QUALITY_SCREENER = 8.0

    def __init__(self, limits: AgentLimits | None = None):
        self.limits = limits or AgentLimits.from_config()

    def propose(self, d: date, features: pd.DataFrame, state: PortfolioState) -> list[TradeIntent]:
        comp = composite_score(features)
        pctl = comp.rank(pct=True)
        held = [s for s, q in state.quantities.items() if q > 0 and s in features.index]

        cash_frac = state.cash_gbp / state.total_value_gbp if state.total_value_gbp > 0 else 0.0
        if cash_frac > self.DEPLOY_CASH_FRAC:
            return self._deploy(features, comp, state)

        intents: list[TradeIntent] = []
        for sym in held:
            if np.isnan(comp.get(sym, np.nan)):
                continue
            row = features.loc[sym]
            w = state.weights.get(sym, 0.0)
            rationale = {"composite": round(float(comp[sym]), 3), "percentile": round(float(pctl[sym]), 3)}

            thesis_sell = bool(row.get("thesis_sell_fired", False))
            below_trend = bool(row.get("dist_sma200", np.nan) < 0)
            if thesis_sell or (pctl[sym] <= self.EXIT_PCTL and below_trend):
                rationale["trigger"] = "thesis sell rule fired" if thesis_sell else "bottom decile and below 200d SMA"
                intents.append(TradeIntent(sym, "exit", None, score=float(comp[sym]) - 1.0, rationale=rationale))
                continue

            hrp = row.get("hrp_weight", np.nan)
            rc = row.get("risk_contribution", np.nan)
            if not np.isnan(hrp) and not np.isnan(rc):
                # hrp/rc are fractions of the held book; rescale to portfolio weights
                invested_frac = sum(state.weights.values())
                hrp_target = float(hrp) * invested_frac
                rc_excess = (float(rc) - float(hrp)) * invested_frac
                if rc_excess > self.RC_EXCESS_TRIM and pctl[sym] < 0.5:
                    target = max(hrp_target, w - self.STEP_WEIGHT)
                    if target < w:
                        rationale["trigger"] = f"risk contribution {rc:.0%} vs HRP {hrp:.0%} of book"
                        intents.append(TradeIntent(sym, "trim", target, score=float(comp[sym]), rationale=rationale))
                    continue
                if pctl[sym] >= self.ADD_PCTL and w < hrp_target - 0.005:
                    target = min(hrp_target, w + self.STEP_WEIGHT, self.limits.max_position_weight)
                    if target > w:
                        rationale["trigger"] = "top quintile, underweight vs HRP target"
                        intents.append(TradeIntent(sym, "add", target, score=float(comp[sym]), rationale=rationale))

        buys = self._new_buys(features, comp, pctl, set(held))
        intents.extend(buys)
        return intents

    def _quality_ok(self, row: pd.Series) -> bool:
        """Tier-B quality gate; passes open when fundamentals are unknown."""
        roic = row.get("roic", np.nan)
        screener = row.get("screener_score", np.nan)
        if np.isnan(roic) and np.isnan(screener):
            return True
        return bool(roic >= self.MIN_QUALITY_ROIC) or bool(screener >= self.MIN_QUALITY_SCREENER)

    def _new_buys(self, features, comp, pctl, held: set) -> list[TradeIntent]:
        candidates = []
        for sym in features.index:
            if sym in held or np.isnan(comp.get(sym, np.nan)):
                continue
            row = features.loc[sym]
            if pctl[sym] < self.BUY_PCTL or not bool(row.get("dist_sma200", np.nan) > 0):
                continue
            if not self._quality_ok(row):
                continue
            candidates.append(sym)
        candidates.sort(key=lambda s: (-comp[s], s))
        return [
            TradeIntent(
                sym,
                "buy",
                self.NEW_POSITION_WEIGHT,
                score=float(comp[sym]),
                rationale={
                    "composite": round(float(comp[sym]), 3),
                    "percentile": round(float(pctl[sym]), 3),
                    "trigger": "top decile, uptrend, quality gate",
                },
            )
            for sym in candidates[: self.MAX_NEW_BUYS]
        ]

    def _deploy(self, features, comp, state: PortfolioState) -> list[TradeIntent]:
        """Build toward an equal-weight book of the top-ranked quality names."""
        ranked = []
        for sym in comp.dropna().sort_values(ascending=False).index:
            row = features.loc[sym]
            if bool(row.get("dist_sma200", np.nan) > 0) and self._quality_ok(row):
                ranked.append(sym)
            if len(ranked) >= self.DEPLOY_POSITIONS:
                break
        if not ranked:
            return []
        target = min(0.9 / len(ranked), self.limits.max_position_weight)
        intents = []
        for sym in ranked:
            w = state.weights.get(sym, 0.0)
            if target > w:
                action = "add" if state.quantities.get(sym, 0) > 0 else "buy"
                intents.append(
                    TradeIntent(
                        sym,
                        action,
                        target,
                        score=float(comp[sym]),
                        rationale={"composite": round(float(comp[sym]), 3), "trigger": "cash deployment"},
                    )
                )
        return intents
