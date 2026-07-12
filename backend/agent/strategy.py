"""Strategy protocol — the swappable "brain" of the trade-suggestion agent.

Implementations (rules, ML, LLM) receive the same inputs live and in backtests;
their output always passes through backend.agent.constraints before it is
persisted or simulated.
"""

from datetime import date
from typing import TYPE_CHECKING, Protocol

from backend.agent.types import PortfolioState, TradeIntent

if TYPE_CHECKING:
    import pandas as pd


class Strategy(Protocol):
    """One decision step: features + portfolio state -> desired trades.

    `features` has one row per symbol (index = yahoo_symbol); columns are the
    price-derived technicals plus FeaturesDaily columns, NaN where a feature is
    unavailable for that symbol/date. Implementations must only use data from
    `features`/`state` — never fetch anything dated after `d`.
    """

    name: str

    def propose(self, d: date, features: "pd.DataFrame", state: PortfolioState) -> list[TradeIntent]: ...
