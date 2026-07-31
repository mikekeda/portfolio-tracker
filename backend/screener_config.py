"""
Screener Configuration System
============================

This module defines all available stock screeners and their criteria in a centralized,
maintainable way. It serves as the single source of truth for screener definitions.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any, Callable, Iterable, Optional, Union


def _is_finite_value(value: Any) -> bool:
    """Check if a value is finite (not None, NaN, or infinite)."""
    if value is None:
        return False

    if isinstance(value, (int, float)):
        return math.isfinite(value)

    # Handle pandas NA and other special values
    try:
        # Try to convert to float and check if finite
        float_val = float(value)
        return math.isfinite(float_val)
    except (ValueError, TypeError):
        # If conversion fails, assume it's not finite
        return False


class ScreenerCategory(Enum):
    """Categories for organizing screeners."""

    FUNDAMENTALS = "fundamentals"
    TECHNICAL = "technical"
    MOMENTUM = "momentum"
    VALUE = "value"
    QUALITY = "quality"
    GROWTH = "growth"
    MACRO_ENVIRONMENT = "macro"  # Added for broader market context screeners


class Sector(str, Enum):
    """Yahoo Finance sector taxonomy. Values must match `info.sector` exactly."""

    TECHNOLOGY = "Technology"
    FINANCIAL_SERVICES = "Financial Services"
    HEALTHCARE = "Healthcare"
    CONSUMER_CYCLICAL = "Consumer Cyclical"
    CONSUMER_DEFENSIVE = "Consumer Defensive"
    INDUSTRIALS = "Industrials"
    COMMUNICATION_SERVICES = "Communication Services"
    ENERGY = "Energy"
    REAL_ESTATE = "Real Estate"
    UTILITIES = "Utilities"
    BASIC_MATERIALS = "Basic Materials"


@dataclass(frozen=True)
class FieldRef:
    """Explicitly marks RHS as a field reference for field-vs-field comparisons."""

    name: str


OperatorFunc = Callable[[Any, Any], bool]

OP_FUNCS: dict[str, OperatorFunc] = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "in": lambda a, b: a in b if isinstance(b, Iterable) and not isinstance(b, (str, bytes)) else False,
    "not_in": lambda a, b: a not in b if isinstance(b, Iterable) and not isinstance(b, (str, bytes)) else False,
}


@dataclass
class ScreenerCriteria:
    """Represents a single criteria for a screener."""

    field: str
    operator: str  # '>=', '<=', '>', '<', '==', '!=', 'in', 'not_in'
    value: Union[Any, FieldRef]  # Can be a literal value or a field reference
    description: str


@dataclass
class ScreenerDefinition:
    """Complete definition of a screener."""

    id: str
    name: str
    description: str
    category: ScreenerCategory
    criteria: list[ScreenerCriteria]
    requires_historical_data: bool = False
    requires_yahoo_data: bool = True
    available: bool = True
    weight: int = 5  # 0..10, usefulness for LT growth/high-risk
    combine_with: list[str] = field(default_factory=list)  # list of screener IDs
    # Sectors where this screener is structurally inappropriate. Empty => all.
    excluded_sectors: frozenset[Sector] = frozenset()
    # Derived at __post_init__ time for O(1) lookup in the combination-bonus hot path.
    combine_with_set: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)
    # Every field the criteria read — two screeners over the same inputs are one
    # signal counted twice, not two confirmations.
    field_set: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)

    def __post_init__(self) -> None:
        self.combine_with_set = frozenset(self.combine_with)

        referenced: set[str] = set()
        for criteria in self.criteria:
            referenced.add(criteria.field)
            if isinstance(criteria.value, FieldRef):
                referenced.add(criteria.value.name)
        self.field_set = frozenset(referenced)


# Quality/FCF/ROIC rules don't apply to banks (leverage by design) or REITs (use FFO, not FCF).
_QUALITY_EXCLUDED: frozenset[Sector] = frozenset({Sector.FINANCIAL_SERVICES, Sector.REAL_ESTATE})
# Debt rules additionally exclude utilities (structural high D/E from regulated rate-base model).
_DEBT_EXCLUDED: frozenset[Sector] = frozenset({Sector.FINANCIAL_SERVICES, Sector.REAL_ESTATE, Sector.UTILITIES})

# Denominator for turning a raw score into a 0..1 ratio. Empirical, not the sum of
# all weights: technical screeners are mutually exclusive, so that sum is unreachable.
SCORE_NORMALIZER = 60
# Above this share of shared inputs, two screeners are one signal counted twice
# rather than independent confirmation, so the pair earns no combination bonus.
MAX_FIELD_OVERLAP = 0.5


class ScreenerConfig:
    """Centralized configuration for all screeners."""

    def __init__(self):
        self.screeners = self._initialize_screeners()
        # Validate configuration on initialization
        self.validate()
        self._applicable_weight = self._compute_applicable_weight()

    def _compute_applicable_weight(self) -> dict[Optional[str], int]:
        """Total positive weight reachable in each sector, keyed by sector name (None = unrestricted)."""
        sectors: list[Optional[str]] = [None, *(s.value for s in Sector)]
        return {
            sector: sum(
                s.weight
                for s in self.screeners.values()
                if s.available and s.weight > 0 and not (sector and sector in s.excluded_sectors)
            )
            for sector in sectors
        }

    def combination_bonus(self, passed_screeners: list[str]) -> int:
        """Bonus for passing screeners that endorse each other *and* read different inputs.

        Diminishing returns (pairs ** 0.85) stop a stock with many passes from
        saturating the score while leaving exceptional ones room to stand out.
        """
        if len(passed_screeners) < 2:
            return 0

        pair_count = 0
        for a, b in combinations(passed_screeners, 2):
            first, second = self.screeners[a], self.screeners[b]
            if b not in first.combine_with_set and a not in second.combine_with_set:
                continue
            smaller = min(len(first.field_set), len(second.field_set))
            if smaller == 0 or len(first.field_set & second.field_set) / smaller > MAX_FIELD_OVERLAP:
                continue
            pair_count += 1

        return round(2 * pair_count**0.85) if pair_count else 0

    def score_normalizer(self, sector: Optional[str]) -> float:
        """Scale a raw score by what its sector could actually earn.

        Financials and REITs are excluded from several screeners, so their raw
        scores are not comparable with the rest of the universe without this.
        """
        unrestricted = self._applicable_weight[None]
        applicable = self._applicable_weight.get(sector, unrestricted)
        return SCORE_NORMALIZER * applicable / unrestricted

    def _initialize_screeners(self) -> dict[str, ScreenerDefinition]:
        """Initialize all available screeners."""
        return {
            # QARP (Quality at Reasonable Price)
            "qarp": ScreenerDefinition(
                id="qarp",
                name="Quality at Reasonable Price",
                description="ROIC >= 8% & PEG <= 2.0 & FCF Margin >= 10%",
                category=ScreenerCategory.FUNDAMENTALS,
                criteria=[
                    ScreenerCriteria("roic", ">=", 8, "Return on Invested Capital >= 8%"),
                    ScreenerCriteria("peg_ratio", "<=", 2.0, "PEG Ratio <= 2.0"),
                    ScreenerCriteria("peg_ratio", ">", 0.0, "PEG Ratio > 0.0"),
                    # Margin, not yield: a yield gate rejects any compounder in a
                    # heavy capex cycle, which is the opposite of the intent.
                    ScreenerCriteria("fcf_margin", ">=", 10, "Free Cash Flow Margin >= 10%"),
                    ScreenerCriteria("profit_margins", ">", 0, "Profitable"),
                ],
                requires_historical_data=False,
                requires_yahoo_data=True,
                available=True,
                weight=7,
                combine_with=[
                    "r40_momentum",
                    "momentum_pullback",
                    "oversold_uptrend",
                    "quality_growth_pullback",
                    "pe_vs_history",
                    "improving_sentiment",
                    "growth_at_reasonable_price",
                    "top_quality_proxy",
                ],
                excluded_sectors=_QUALITY_EXCLUDED,
            ),
            # Value + Quality
            "value_quality": ScreenerDefinition(
                id="value_quality",
                name="Value + Quality",
                description="PE <= 20 & ROIC >= 8% & Revenue Growth > 0% & Profit Margins > 0",
                category=ScreenerCategory.VALUE,
                criteria=[
                    ScreenerCriteria("pe_ratio", "<=", 20, "PE Ratio <= 20"),
                    ScreenerCriteria("pe_ratio", ">", 0, "PE Ratio > 0"),
                    ScreenerCriteria("roic", ">=", 8, "Return on Invested Capital >= 8%"),
                    ScreenerCriteria("revenue_growth", ">", 0, "Revenue Growth > 0%"),
                    ScreenerCriteria("profit_margins", ">", 0, "Profitable"),
                ],
                requires_historical_data=False,
                requires_yahoo_data=True,
                available=True,
                weight=4,
                combine_with=[
                    "golden_cross",
                    "oversold_uptrend",
                    "momentum_pullback",
                    "pe_compression",
                    "pe_vs_history",
                    "top_quality_proxy",
                ],
                excluded_sectors=_QUALITY_EXCLUDED,
            ),
            # Growth at Reasonable Price
            "growth_at_reasonable_price": ScreenerDefinition(
                id="growth_at_reasonable_price",
                name="Growth at Reasonable Price",
                description="Revenue Growth >= 10% & PEG <= 1.5 & Profit Margins > 5%",
                category=ScreenerCategory.GROWTH,
                criteria=[
                    ScreenerCriteria("revenue_growth", ">=", 10, "Revenue Growth >= 10%"),
                    ScreenerCriteria("peg_ratio", "<=", 1.5, "PEG Ratio <= 1.5"),
                    ScreenerCriteria("peg_ratio", ">", 0.0, "PEG Ratio > 0"),
                    ScreenerCriteria("pe_ratio", ">", 0, "Positive PE (hygiene)"),
                    ScreenerCriteria("profit_margins", ">", 5, "Profit Margins > 5%"),
                ],
                requires_historical_data=False,
                requires_yahoo_data=True,
                available=True,
                weight=6,
                combine_with=[
                    "r40_momentum",
                    "momentum_pullback",
                    "breakout_quiet_base",
                    "oversold_uptrend",
                    "qarp",
                    "pe_vs_history",
                    "top_quality_proxy",
                ],
                excluded_sectors=_QUALITY_EXCLUDED,
            ),
            # Oversold in Uptrend
            "oversold_uptrend": ScreenerDefinition(
                id="oversold_uptrend",
                name="Oversold in Uptrend",
                description="RSI <= 35 & Price > SMA(200) & Price <= SMA(20)",
                category=ScreenerCategory.TECHNICAL,
                criteria=[
                    ScreenerCriteria("rsi", "<=", 35, "RSI <= 35 (Oversold)"),
                    ScreenerCriteria("current_price", ">", FieldRef("sma_200"), "Price > SMA(200) (Uptrend)"),
                    ScreenerCriteria("current_price", "<=", FieldRef("sma_20"), "Price <= SMA(20) (Pullback)"),
                ],
                requires_historical_data=True,
                requires_yahoo_data=True,
                available=True,
                weight=5,
                combine_with=[
                    "r40_momentum",
                    "qarp",
                    "growth_at_reasonable_price",
                    "quality_growth_pullback",
                    "improving_sentiment",
                    "value_quality",
                    "top_quality_proxy",
                ],
            ),
            # High Short Interest
            "high_short_interest": ScreenerDefinition(
                # id is stable — renaming it would orphan historical passed_screeners.
                id="high_short_interest",
                name="Short Squeeze Setup",
                description="Short Float >= 10% AND the price is reclaiming both moving averages on accumulation volume. High short interest alone does not qualify — in a downtrend it means the shorts are right.",
                category=ScreenerCategory.MOMENTUM,
                criteria=[
                    ScreenerCriteria("short_percent_of_float", ">=", 10, "High short interest"),
                    ScreenerCriteria("current_price", ">", FieldRef("sma_50"), "Reclaiming SMA50"),
                    ScreenerCriteria("current_price", ">", FieldRef("sma_200"), "Primary uptrend"),
                    ScreenerCriteria("volume_ratio", ">=", 1.5, "Accumulation"),
                ],
                requires_historical_data=True,
                requires_yahoo_data=True,
                available=True,
                # Deliberately low: squeeze setups are drill-down candidates, not a
                # quality endorsement — short interest >= 10% cuts both ways.
                weight=1,
                combine_with=["golden_cross", "breakout_quiet_base", "momentum_pullback"],
            ),
            # Momentum Pullback
            "momentum_pullback": ScreenerDefinition(
                id="momentum_pullback",
                name="Momentum Pullback",
                description="RS(6m) >= +10pp vs SPY & Close <= SMA20 & RSI 35–50",
                category=ScreenerCategory.MOMENTUM,
                criteria=[
                    ScreenerCriteria("rs_6m_vs_spy", ">=", 10, "RS >= +10pp over 6m"),
                    ScreenerCriteria("current_price", ">", FieldRef("sma_200"), "Primary uptrend"),
                    ScreenerCriteria("current_price", "<=", FieldRef("sma_20"), "Pullback to SMA20"),
                    ScreenerCriteria("rsi", ">=", 35, "RSI >= 35"),
                    ScreenerCriteria("rsi", "<=", 50, "RSI <= 50"),
                ],
                requires_historical_data=True,
                requires_yahoo_data=True,
                available=True,
                weight=6,
                combine_with=[
                    "r40_momentum",
                    "qarp",
                    "growth_at_reasonable_price",
                    "improving_sentiment",
                    "value_quality",
                    "high_short_interest",
                    "top_quality_proxy",
                    "pe_compression",
                    "golden_cross",
                ],
            ),
            # Golden Cross + First Pullback
            "golden_cross": ScreenerDefinition(
                id="golden_cross",
                name="Golden Cross + First Pullback",
                description="SMA(50) crossed above SMA(200) within 60d & Close within -2%..+1% of SMA(50)",
                category=ScreenerCategory.TECHNICAL,
                criteria=[
                    ScreenerCriteria("gc_days_since", "<=", 60, "SMA50/200 crossed within 60d"),
                    ScreenerCriteria("gc_days_since", ">=", 0, "Cross exists"),
                    # gc_days_since fires on crosses in BOTH directions; without this
                    # direction check a recent death cross would also pass.
                    ScreenerCriteria("sma_50", ">", FieldRef("sma_200"), "Cross was bullish (SMA50 above SMA200)"),
                    ScreenerCriteria("gc_within_sma50_frac", ">=", -0.02, "Close near SMA50"),
                    ScreenerCriteria("gc_within_sma50_frac", "<=", 0.01, "Close not extended past SMA50"),
                ],
                requires_historical_data=True,
                requires_yahoo_data=True,
                available=True,
                weight=4,
                combine_with=[
                    "momentum_pullback",
                    "r40_momentum",
                    "quality_growth_pullback",
                    "value_quality",
                    "high_short_interest",
                    "pe_compression",
                ],
            ),
            # Volatility Contraction Pattern
            "volatility_contraction": ScreenerDefinition(
                id="volatility_contraction",
                name="Volatility Contraction Pattern",
                description="BBWidth(20) <= 30th pct of 6m & 20d vol < 60d vol",
                category=ScreenerCategory.TECHNICAL,
                criteria=[
                    ScreenerCriteria("bb_width_20", "<=", FieldRef("bb_width_20_p30_6m"), "Tight vs 6m history"),
                    ScreenerCriteria("vol20_lt_vol60", "==", True, "Volume contraction"),
                    ScreenerCriteria("rs_6m_vs_spy", ">=", 10, "Leading the market by >=10pp"),
                ],
                requires_historical_data=True,
                requires_yahoo_data=True,
                available=True,
                weight=4,
                combine_with=["breakout_quiet_base", "r40_momentum"],
            ),
            # Rule-of-40 + Momentum
            "r40_momentum": ScreenerDefinition(
                id="r40_momentum",
                name="Rule of 40 + Momentum",
                description="Growth quality with trend confirmation",
                category=ScreenerCategory.GROWTH,
                criteria=[
                    ScreenerCriteria("rule_of_40_score", ">=", 40, "Rule of 40"),
                    ScreenerCriteria("current_price", ">=", FieldRef("sma_50"), "Above 50DMA"),
                    ScreenerCriteria("rs_6m_vs_spy", ">=", 10, "Leader"),
                ],
                requires_historical_data=True,
                requires_yahoo_data=True,
                available=True,
                weight=9,
                combine_with=[
                    "qarp",
                    "growth_at_reasonable_price",
                    "oversold_uptrend",
                    "momentum_pullback",
                    "quality_growth_pullback",
                    "breakout_quiet_base",
                    "pe_vs_history",
                    "volatility_contraction",
                    "golden_cross",
                    "pe_compression",
                    "top_quality_proxy",
                ],
                excluded_sectors=_DEBT_EXCLUDED,
            ),
            # 52-Week Breakout (Quiet Base + Volume)
            "breakout_quiet_base": ScreenerDefinition(
                id="breakout_quiet_base",
                name="52W Breakout (Quiet Base + Volume)",
                description="Near 52W high with tightness and confirming volume",
                category=ScreenerCategory.MOMENTUM,
                criteria=[
                    # The field is (price - high)/high, so it is <= 0 by construction.
                    ScreenerCriteria("fifty_two_week_high_distance", ">=", -2.5, "Within 2.5% below 52W high"),
                    ScreenerCriteria("bb_width_20", "<=", FieldRef("bb_width_20_p30_6m"), "Tight vs 6m"),
                    ScreenerCriteria("volume_ratio", ">=", 1.3, "Volume confirmation"),
                    ScreenerCriteria("rs_6m_vs_spy", ">=", 10, "Leader"),
                ],
                requires_historical_data=True,
                requires_yahoo_data=True,
                available=True,
                weight=6,
                combine_with=[
                    "volatility_contraction",
                    "r40_momentum",
                    "growth_at_reasonable_price",
                    "high_short_interest",
                ],
            ),
            # Quality Growth Pullback
            "quality_growth_pullback": ScreenerDefinition(
                id="quality_growth_pullback",
                name="Quality Growth Pullback",
                description="Quality/profitability with controlled dip to 50DMA",
                category=ScreenerCategory.QUALITY,
                criteria=[
                    ScreenerCriteria("roic", ">=", 12, "Quality ROIC >= 12%"),
                    ScreenerCriteria("profit_margins", ">=", 10, "Profitability"),
                    ScreenerCriteria("current_price", "<=", FieldRef("sma_50"), "At/below 50DMA"),
                    ScreenerCriteria("current_price", ">=", FieldRef("sma_200"), "Primary uptrend"),
                    ScreenerCriteria("rsi", "<=", 55, "Not overheated"),
                ],
                requires_historical_data=True,
                requires_yahoo_data=True,
                available=True,
                weight=8,
                combine_with=[
                    "r40_momentum",
                    "qarp",
                    "golden_cross",
                    "pe_compression",
                    "oversold_uptrend",
                    "top_quality_proxy",
                ],
                excluded_sectors=_QUALITY_EXCLUDED,
            ),
            # Improving Analyst Sentiment
            "improving_sentiment": ScreenerDefinition(
                id="improving_sentiment",
                name="Improving Analyst Sentiment",
                description="Analyst sentiment has been steadily improving over recent months while PE remains reasonable.",
                category=ScreenerCategory.FUNDAMENTALS,
                criteria=[
                    # recommendation_trend is a correlation in [-1, 1] (see
                    # calculate_historical_trends); 0.5 = consistent improvement.
                    ScreenerCriteria("recommendation_trend", ">=", 0.5, "Sentiment trend consistently positive"),
                    # The correlation alone passes a drift from 0.70 to 0.71.
                    ScreenerCriteria("recommendation_delta_12m", ">=", 0.05, "Improvement is large enough to matter"),
                    ScreenerCriteria("pe_ratio", "<=", 40, "Valuation is not excessive"),
                    ScreenerCriteria("pe_ratio", ">", 0, "Company is profitable"),
                ],
                requires_historical_data=True,  # Requires recommendation history
                requires_yahoo_data=True,
                available=True,
                weight=4,
                combine_with=["qarp", "momentum_pullback", "oversold_uptrend", "pe_vs_history"],
            ),
            # PE Compression
            "pe_compression": ScreenerDefinition(
                id="pe_compression",
                name="PE Compression (Earnings > Price)",
                description="The P/E ratio has declined over the last year as earnings grew faster than the stock price.",
                category=ScreenerCategory.VALUE,
                criteria=[
                    ScreenerCriteria("pe_1y_trend_pct", "<=", -10, "P/E ratio has fallen by at least 10%"),
                    ScreenerCriteria("revenue_growth", ">=", 5, "Still a growing business (not a value trap)"),
                    ScreenerCriteria("pe_ratio", ">", 0, "Company is profitable"),
                ],
                requires_historical_data=True,  # Requires PE history
                requires_yahoo_data=True,
                available=True,
                weight=5,
                combine_with=[
                    "quality_growth_pullback",
                    "golden_cross",
                    "r40_momentum",
                    "momentum_pullback",
                    "value_quality",
                ],
            ),
            # Insider Cluster Buying (Lakonishok-Lee).
            # Yahoo proxies SEC Form 3/4, so coverage is effectively US-only — UK/EU
            # holdings will simply not pass this rule (insider_buy_count_90d=0).
            "insider_buying": ScreenerDefinition(
                id="insider_buying",
                name="Insider Cluster Buying",
                description=">= 2 distinct insiders net-bought in last 90d with net value >= $50k",
                category=ScreenerCategory.FUNDAMENTALS,
                criteria=[
                    ScreenerCriteria("insider_buy_count_90d", ">=", 2, ">= 2 distinct insider buyers in 90d"),
                    ScreenerCriteria("insider_net_value_90d", ">=", 50_000, "Net buy >= $50k (filters token trades)"),
                ],
                requires_historical_data=False,
                requires_yahoo_data=False,
                available=True,
                weight=6,
                combine_with=[
                    "qarp",
                    "top_quality_proxy",
                    "pe_vs_history",
                    "value_quality",
                    "growth_at_reasonable_price",
                    "oversold_uptrend",
                    "momentum_pullback",
                    "quality_growth_pullback",
                ],
            ),
            "pe_vs_history": ScreenerDefinition(
                id="pe_vs_history",
                name="PE Below 5-Year Average",
                description="The current P/E ratio is trading below its own 5-year historical average.",
                category=ScreenerCategory.VALUE,
                criteria=[
                    ScreenerCriteria("pe_5y_avg_vs_current_pct", ">=", 10, "Current PE is at least 10% below 5Y avg"),
                    ScreenerCriteria("roic", ">=", 8, "Business quality remains high (ROIC >= 8%)"),
                    ScreenerCriteria("pe_ratio", ">", 0, "Company is profitable"),
                ],
                requires_historical_data=True,  # Requires PE history
                requires_yahoo_data=True,
                available=True,
                # Mean-reversion on a multiple, and the current-vs-history basis
                # gap is wide enough that this cannot carry a top weight.
                weight=5,
                combine_with=[
                    "r40_momentum",
                    "qarp",
                    "improving_sentiment",
                    "top_quality_proxy",
                    "growth_at_reasonable_price",
                    "value_quality",
                ],
                excluded_sectors=_QUALITY_EXCLUDED,
            ),
            # Negative screeners
            "death_cross": ScreenerDefinition(
                id="death_cross",
                name="Bearish: Death Cross",
                description="SMA(50) crossed below SMA(200) within 60d, indicating a major trend change to the downside.",
                category=ScreenerCategory.MACRO_ENVIRONMENT,
                criteria=[
                    # Note: We can reuse gc_days_since; it just tells us days since *any* cross.
                    # The second criterion confirms the direction of the cross was bearish.
                    ScreenerCriteria("gc_days_since", "<=", 60, "SMA50/200 crossed within 60d"),
                    ScreenerCriteria("gc_days_since", ">=", 0, "Cross exists"),
                    ScreenerCriteria("sma_50", "<", FieldRef("sma_200"), "SMA50 is currently below SMA200"),
                ],
                requires_historical_data=True,
                requires_yahoo_data=True,
                available=True,  # Disabled by default
                weight=-2,  # Short-term technical signal, shouldn't dominate the composite
                combine_with=[],
            ),
            "red_flag_low_roic": ScreenerDefinition(
                id="red_flag_low_roic",
                name="Red Flag: Low ROIC",
                description="""Company generates poor returns on invested capital (< 5%). High-growth pre-profit companies are exempt — low ROIC is the expected state, not a red flag.""",
                category=ScreenerCategory.QUALITY,
                criteria=[
                    ScreenerCriteria("roic", "<", 5, "ROIC is very low (< 5%)"),
                    ScreenerCriteria("revenue_growth", "<", 25, "Not high-growth (low ROIC is not a scaling phase)"),
                ],
                available=True,
                weight=-4,  # Apply a partial penalty
                excluded_sectors=_QUALITY_EXCLUDED,
            ),
            "red_flag_high_debt": ScreenerDefinition(
                id="red_flag_high_debt",
                name="Red Flag: High Debt",
                description="""Company has excessive debt burden (net debt > 3x EBITDA).""",
                category=ScreenerCategory.QUALITY,
                criteria=[
                    ScreenerCriteria("net_debt_to_ebitda", ">", 3.0, "Net debt exceeds 3x EBITDA"),
                ],
                available=True,
                weight=-3,
                excluded_sectors=_DEBT_EXCLUDED,
            ),
            "red_flag_cash_burn": ScreenerDefinition(
                id="red_flag_cash_burn",
                name="Red Flag: Negative FCF",
                description="""Mature company burning cash (Negative FCF, Revenue Growth < 30%, Revenue >= $100M). Pre-revenue biotechs, small commercial-stage names, and high-growth pre-profit companies are exempt.""",
                category=ScreenerCategory.QUALITY,
                criteria=[
                    ScreenerCriteria("free_cashflow_yield", "<", 0, "Company is burning cash"),
                    ScreenerCriteria("revenue_growth", "<", 30, "Not high-growth (burn is not intentional investment)"),
                    # Excludes pre-revenue biotechs. Yahoo totalRevenue is in reporting ccy.
                    ScreenerCriteria("total_revenue", ">=", 100_000_000, "Has meaningful revenue (>= $100M, not pre-commercial)"),
                ],
                available=True,
                weight=-5,  # This is a significant red flag
                excluded_sectors=_QUALITY_EXCLUDED,
            ),
            # Top Quality (based on screenshot)
            "top_quality_proxy": ScreenerDefinition(
                id="top_quality_proxy",
                name="Tom Nash screener",
                description="""Strong TTM growth, profitability, and balance sheet as a proxy for consistent quality.
Check those criteria:
1. Positive free cash flow 4 to 6 quarters
2. Net cash positive 4 to 6 quarters
3. Revenue growth at 10% or better
4. Clear moat
5. CEO and team with track record
6. No major legal or regulatory overhang""",
                category=ScreenerCategory.QUALITY,
                criteria=[
                    ScreenerCriteria("revenue_growth", ">=", 10, "Revenue Growth >= 10% (TTM)"),
                    # Proxy for "positive free cash flow 4 to 6 quarters".
                    ScreenerCriteria("fcf_margin", ">=", 8, "Positive FCF Margin >= 8% (TTM)"),
                    # Proxy for "net cash positive". Debt/equity is distorted by
                    # buybacks the same way ROE is; leverage vs earnings is not.
                    ScreenerCriteria("net_debt_to_ebitda", "<=", 1.0, "Net debt <= 1x EBITDA"),
                    # These are proxies for "Clear moat"
                    ScreenerCriteria("roic", ">=", 10, "High ROIC (Moat proxy) >= 10%"),
                    ScreenerCriteria("profit_margins", ">=", 10, "High Profit Margins (Moat proxy) >= 10%"),
                ],
                requires_historical_data=False,  # We are using current/TTM data
                requires_yahoo_data=True,
                available=True,
                weight=10,  # This is a very strong, high-quality signal
                combine_with=[
                    "momentum_pullback",
                    "oversold_uptrend",
                    "pe_vs_history",
                    "r40_momentum",
                    "quality_growth_pullback",
                    "qarp",
                    "value_quality",
                    "growth_at_reasonable_price",
                ],
                excluded_sectors=_DEBT_EXCLUDED,
                # Combines well with entry points and other quality/growth validators
            ),
            # Durable compounding — the only screener that looks past a single TTM snapshot.
            "roic_consistency": ScreenerDefinition(
                id="roic_consistency",
                name="Consistent High ROIC",
                description="ROIC >= 12% in each of the last 3 fiscal years — separates durable compounders from cyclical rebounds that flatter a single TTM window.",
                category=ScreenerCategory.QUALITY,
                criteria=[
                    ScreenerCriteria("roic_3y_min", ">=", 12, "ROIC >= 12% in every one of the last 3 years"),
                    ScreenerCriteria("revenue_growth", ">", 0, "Still growing"),
                ],
                requires_historical_data=False,
                requires_yahoo_data=True,
                available=True,
                weight=10,
                combine_with=[
                    "r40_momentum",
                    "growth_at_reasonable_price",
                    "momentum_pullback",
                    "oversold_uptrend",
                    "quality_growth_pullback",
                    "breakout_quiet_base",
                    "insider_buying",
                ],
                excluded_sectors=_QUALITY_EXCLUDED,
            ),
            # Pre-profit growth: every other positive screener needs a positive
            # P/E, PEG, FCF or ROIC, so without this the sleeve can only score negative.
            "pre_profit_growth": ScreenerDefinition(
                id="pre_profit_growth",
                name="Funded Pre-Profit Growth",
                description="Revenue Growth >= 25% & Gross Margin >= 30% & net cash & above 200DMA. Gross margin is the test of whether the unit economics can eventually carry fixed costs; net cash is the test of whether it survives to find out.",
                category=ScreenerCategory.GROWTH,
                criteria=[
                    # Without this a profitable compounder scores here as well as
                    # on the quality screeners, double-counting the same growth.
                    ScreenerCriteria("profit_margins", "<=", 0, "Not yet profitable"),
                    ScreenerCriteria("revenue_growth", ">=", 25, "Revenue Growth >= 25%"),
                    ScreenerCriteria("gross_margin", ">=", 30, "Gross Margin >= 30% (viable unit economics)"),
                    ScreenerCriteria("net_cash", "==", True, "Net cash (funded through to profitability)"),
                    ScreenerCriteria("current_price", ">", FieldRef("sma_200"), "Primary uptrend"),
                ],
                requires_historical_data=True,
                requires_yahoo_data=True,
                available=True,
                weight=6,
                combine_with=[
                    "r40_momentum",
                    "breakout_quiet_base",
                    "momentum_pullback",
                    "volatility_contraction",
                    "insider_buying",
                ],
                excluded_sectors=_QUALITY_EXCLUDED,
            ),
        }

    def get_available_screeners(self) -> list[ScreenerDefinition]:
        """Get only available screeners."""
        return [s for s in self.screeners.values() if s.available]

    def get_available_operators(self) -> list[str]:
        """Get list of available operators."""
        return [">=", "<=", ">", "<", "==", "!=", "in", "not_in"]

    def get_available_fields(self) -> list[str]:
        """Get list of available field names."""
        return [
            "free_cashflow_yield",
            "fcf_margin",
            "peg_ratio",
            "revenue_growth",
            "profit_margins",
            "gross_margin",
            "pe_ratio",
            "short_percent_of_float",
            "rsi",
            "current_price",
            "rule_of_40_score",
            "sma_20",
            "sma_50",
            "sma_200",
            "volume_ratio",
            "bb_width_20",
            "bb_width_20_p30_6m",
            "gc_days_since",
            "gc_within_sma50_frac",
            "vol20_lt_vol60",
            "rs_6m_vs_spy",
            "fifty_two_week_high_distance",
            "recommendation_trend",
            "recommendation_delta_12m",
            "pe_1y_trend_pct",
            "pe_5y_avg_vs_current_pct",
            "roic",
            "roic_3y_min",
            "net_cash",
            "net_debt_to_ebitda",
            "total_revenue",
            "insider_buy_count_90d",
            "insider_net_value_90d",
        ]

    def validate(self) -> None:
        """Validate screener configuration for errors."""
        available_fields = set(self.get_available_fields())
        available_operators = set(self.get_available_operators())

        for screener_id, screener_def in self.screeners.items():
            for criteria in screener_def.criteria:
                # Validate field name
                if criteria.field not in available_fields:
                    raise ValueError(f"Screener '{screener_id}': Unknown field '{criteria.field}'")

                # Validate operator
                if criteria.operator not in available_operators:
                    raise ValueError(f"Screener '{screener_id}': Unknown operator '{criteria.operator}'")

                # Validate field reference
                if isinstance(criteria.value, FieldRef):
                    if criteria.value.name not in available_fields:
                        raise ValueError(f"Screener '{screener_id}': Unknown field reference '{criteria.value.name}'")

    def eval_criterion(self, fields: dict[str, Any], criteria: ScreenerCriteria) -> tuple[bool, str]:
        """Evaluate a single criterion. Operators and field names are assumed valid;
        `validate()` enforces that at startup so the hot path stays check-free."""
        lhs = fields.get(criteria.field)
        if not _is_finite_value(lhs):
            return False, f"field {criteria.field} is None or non-finite"

        if isinstance(criteria.value, FieldRef):
            rhs = fields.get(criteria.value.name)
            if not _is_finite_value(rhs):
                return False, f"field {criteria.value.name} is None or non-finite"
        else:
            rhs = criteria.value

        ok = OP_FUNCS[criteria.operator](lhs, rhs)
        return ok, (criteria.description or f"{criteria.field} {criteria.operator} {criteria.value}")

    def passes_screener(self, fields: dict[str, Any], screener_def: ScreenerDefinition) -> bool:
        """Short-circuiting boolean check used by the screener hot loop. Returns False
        as soon as any criterion fails, without allocating a details list."""
        if screener_def.excluded_sectors:
            sector = fields.get("sector")
            if sector and sector in screener_def.excluded_sectors:
                return False
        for criteria in screener_def.criteria:
            ok, _ = self.eval_criterion(fields, criteria)
            if not ok:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert only available screeners to dictionary format for API responses."""
        return {
            "screeners": [
                {
                    "id": screener.id,
                    "name": screener.name,
                    "description": screener.description,
                    "category": screener.category.value,
                    "criteria": [
                        {
                            "field": criteria.field,
                            "operator": criteria.operator,
                            "value": {"fieldRef": criteria.value.name}
                            if isinstance(criteria.value, FieldRef)
                            else criteria.value,
                            "description": criteria.description,
                        }
                        for criteria in screener.criteria
                    ],
                    "requires_historical_data": screener.requires_historical_data,
                    "requires_yahoo_data": screener.requires_yahoo_data,
                    "available": screener.available,
                    "weight": screener.weight,
                    "combine_with": screener.combine_with,
                    "excluded_sectors": sorted(s.value for s in screener.excluded_sectors),
                }
                for screener in self.screeners.values()
                if screener.available
            ]
        }


# Global instance
_screener_config = None


def get_screener_config() -> ScreenerConfig:
    """Get the global screener configuration instance."""
    global _screener_config
    if _screener_config is None:
        _screener_config = ScreenerConfig()
    return _screener_config
