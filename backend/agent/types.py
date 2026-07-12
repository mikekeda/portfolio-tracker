"""Shared datatypes for the trade-suggestion agent.

Pure stdlib on purpose: the constraint layer and its tests must be runnable
without the app's config (which probes cloud metadata endpoints on import).
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Optional

Action = Literal["buy", "add", "trim", "exit"]

# Currencies already denominated in sterling — no T212 FX fee applies.
STERLING = frozenset({"GBP", "GBX", "GBp"})


@dataclass(frozen=True)
class TradeIntent:
    """A strategy's desired trade, before safety caps are applied."""

    symbol: str  # yahoo_symbol
    action: Action
    # Desired end weight (fraction of portfolio). Required for buy/add/trim;
    # ignored for exit (an exit always closes the full position).
    target_weight: Optional[float]
    # Strategy conviction; |score| ranks intents when the turnover budget binds.
    score: float
    # JSONB-able explanation: which signals fired and their values.
    rationale: dict = field(default_factory=dict)


@dataclass
class PortfolioState:
    """Snapshot of the portfolio the agent is trading against."""

    date: date
    total_value_gbp: float
    cash_gbp: float
    weights: dict[str, float]  # symbol -> current weight (fraction, held only)
    quantities: dict[str, float]  # symbol -> shares held
    currencies: dict[str, str]  # symbol -> native currency (FX-fee awareness)
    tags: dict[str, list[str]] = field(default_factory=dict)  # symbol -> instrument tags

    @property
    def held_count(self) -> int:
        return sum(1 for q in self.quantities.values() if q > 0)


@dataclass(frozen=True)
class TradeOrder:
    """A constraint-checked order ready to persist or simulate.

    value_gbp == 0 means the intent was fully vetoed; `adjustments` says why.
    Callers that execute orders should filter on `executable`.
    """

    symbol: str
    action: Action
    value_gbp: float
    quantity: float
    weight_before: float
    weight_after: float
    score: float
    fee_gbp: float
    rationale: dict
    adjustments: tuple[str, ...] = ()

    @property
    def executable(self) -> bool:
        return self.value_gbp > 0


@dataclass(frozen=True)
class AgentLimits:
    """Deterministic safety caps applied to every strategy's output."""

    max_daily_turnover: float  # fraction of portfolio value, buys + sells
    min_holdings: int
    max_position_weight: float  # fraction
    min_trade_gbp: float
    fx_fee: float  # fraction of non-GBP notional
    cluster_caps: dict[str, float] = field(default_factory=dict)  # tag -> cap in percent

    @classmethod
    def from_config(cls) -> "AgentLimits":
        import config

        return cls(
            max_daily_turnover=config.AGENT_MAX_DAILY_TURNOVER,
            min_holdings=config.AGENT_MIN_HOLDINGS,
            max_position_weight=config.AGENT_MAX_POSITION_WEIGHT,
            min_trade_gbp=config.AGENT_MIN_TRADE_GBP,
            fx_fee=config.T212_FX_FEE,
            cluster_caps=dict(config.TAG_CLUSTER_SOFT_CAPS),
        )
