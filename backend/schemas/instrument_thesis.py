"""User-authored thesis JSON stored on Instrument.thesis (JSONB)."""

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

ThesisConviction = Literal["low", "medium", "high"]

# Holdings dict keys plus thesis band keys injected at eval time.
THESIS_RULE_FIELDS = frozenset({
    "return_on_equity",
    "return_on_assets",
    "free_cashflow_yield",
    "peg_ratio",
    "revenue_growth",
    "profit_margins",
    "pe_ratio",
    "forward_pe_ratio",
    "short_percent_of_float",
    "fifty_two_week_change",
    "fifty_two_week_high_distance",
    "rsi",
    "institutional_ownership",
    "current_price",
    "rule_of_40_score",
    "portfolio_pct",
    "market_value",
    "return_pct",
    "roic",
    "ps_ratio",
    "beta",
    "quantity",
    "target_weight_min_pct",
    "target_weight_max_pct",
})

THESIS_RULE_OPERATORS = frozenset({">=", "<=", ">", "<", "==", "!=", "in", "not_in"})


class ThesisRuleSchema(BaseModel):
    """Single screener-style rule for thesis buy_rules / sell_rules."""

    model_config = ConfigDict(extra="forbid")

    field: str
    operator: str
    value: Union[float, int, str, dict[str, str]]
    description: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_operator_alias(cls, data: Any) -> Any:
        if isinstance(data, dict) and "op" in data and "operator" not in data:
            data = {**data, "operator": data["op"]}
        return data

    @model_validator(mode="after")
    def validate_rule(self) -> "ThesisRuleSchema":
        if self.field not in THESIS_RULE_FIELDS:
            raise ValueError(f"field must be one of: {sorted(THESIS_RULE_FIELDS)}")
        if self.operator not in THESIS_RULE_OPERATORS:
            raise ValueError(f"operator must be one of: {sorted(THESIS_RULE_OPERATORS)}")
        return self


class InstrumentThesisSchema(BaseModel):
    """Validated thesis document on Instrument.thesis."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    target_weight_min_pct: float = Field(
        ge=0.0,
        le=100.0,
        description="Minimum target allocation as % of total portfolio (0–100)",
    )
    target_weight_max_pct: float = Field(
        ge=0.0,
        le=100.0,
        description="Maximum target allocation as % of total portfolio (0–100)",
    )
    buy_rules: list[ThesisRuleSchema] = Field(default_factory=list)
    sell_rules: list[ThesisRuleSchema] = Field(default_factory=list)
    buy_triggers: list[str] = Field(default_factory=list)
    sell_triggers: list[str] = Field(default_factory=list)
    horizon_years: int = Field(ge=1, le=100)
    conviction: ThesisConviction | None = None

    @model_validator(mode="after")
    def target_weight_band_ordered(self) -> "InstrumentThesisSchema":
        if self.target_weight_min_pct > self.target_weight_max_pct:
            raise ValueError("target_weight_min_pct must be <= target_weight_max_pct")
        return self
