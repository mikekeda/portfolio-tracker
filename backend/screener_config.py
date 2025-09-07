"""
Screener Configuration System
============================

This module defines all available stock screeners and their criteria in a centralized,
maintainable way. It serves as the single source of truth for screener definitions.

HOW TO ADD A NEW SCREENER:
==========================

1. Add a new ScreenerDefinition in the _initialize_screeners() method:

   screeners['new_screener_id'] = ScreenerDefinition(
       id='new_screener_id',
       name='New Screener Name',
       description='Brief description of what this screener does',
       category=ScreenerCategory.FUNDAMENTALS,  # or TECHNICAL, MOMENTUM, VALUE, QUALITY, GROWTH
       criteria=[
           ScreenerCriteria('field_name', '>=', 10, 'Description of this criteria'),
           ScreenerCriteria('another_field', '<=', 2.0, 'Another criteria description'),
       ],
       requires_historical_data=False,  # Set to True if you need price history data
       requires_yahoo_data=True,        # Set to False if you don't need Yahoo Finance data
       available=True                   # Set to False to disable the screener
   )

2. Available field names (use exact names from portfolio API):
   - return_on_equity, return_on_assets, free_cashflow_yield
   - peg_ratio, revenue_growth, profit_margins, pe_ratio
   - short_percent_of_float, fifty_two_week_change, rsi
   - institutional_ownership, current_price, sma_20, sma_50, sma_200, volume_ratio

3. Available operators: >=, <=, >, <, ==, !=

4. Available categories: FUNDAMENTALS, TECHNICAL, MOMENTUM, VALUE, QUALITY, GROWTH

5. The screener will automatically work in both backend and frontend - no additional code needed!

HOW TO MODIFY AN EXISTING SCREENER:
===================================

Simply update the criteria, values, or description in the ScreenerDefinition.
Changes will automatically apply to both backend and frontend.

FIELD MAPPING:
==============

The configuration uses field names that match the portfolio API response:
- Frontend: Uses field names directly from portfolio API
- Backend: Maps field names to internal metrics via _get_field_value()

This ensures consistency and eliminates the need for complex field mapping.
"""

from typing import Dict, List, Any, Union, Callable, Iterable
from dataclasses import dataclass
from enum import Enum
import math


def _is_finite_value(value: Any) -> bool:
    """
    Check if a value is finite (not None, NaN, or infinite).

    Args:
        value: The value to check

    Returns:
        True if the value is finite, False otherwise
    """
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


@dataclass(frozen=True)
class FieldRef:
    """Explicitly marks RHS as a field reference for field-vs-field comparisons."""
    name: str


OperatorFunc = Callable[[Any, Any], bool]

OP_FUNCS: Dict[str, OperatorFunc] = {
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
    criteria: List[ScreenerCriteria]
    requires_historical_data: bool = False
    requires_yahoo_data: bool = True
    available: bool = True


class ScreenerConfig:
    """Centralized configuration for all screeners."""

    def __init__(self):
        self._screeners = self._initialize_screeners()
        # Validate configuration on initialization
        self.validate()

    def _initialize_screeners(self) -> Dict[str, ScreenerDefinition]:
        """Initialize all available screeners."""
        return {
            # QARP (Quality at Reasonable Price)
            'qarp': ScreenerDefinition(
                id='qarp',
                name='Quality at Reasonable Price',
                description='ROE >= 10% & PEG <= 2.0 & FCF Yield >= 4%',
                category=ScreenerCategory.FUNDAMENTALS,
                criteria=[
                    ScreenerCriteria('return_on_equity', '>=', 10, 'Return on Equity >= 10%'),
                    ScreenerCriteria('peg_ratio', '<=', 2.0, 'PEG Ratio <= 2.0'),
                    ScreenerCriteria('peg_ratio', '>', 0.0, 'PEG Ratio > 0.0'),
                    ScreenerCriteria('free_cashflow_yield', '>=', 4, 'Free Cash Flow Yield >= 4%'),
                    ScreenerCriteria('profit_margins', '>', 0, 'Profitable'),
                ],
                requires_historical_data=False,
                requires_yahoo_data=True,
                available=True
            ),

            # Value + Quality
            'value_quality': ScreenerDefinition(
                id='value_quality',
                name='Value + Quality',
                description='PE <= 20 & ROA >= 5% & Revenue Growth > 0% & Profit Margins > 0',
                category=ScreenerCategory.VALUE,
                criteria=[
                    ScreenerCriteria('pe_ratio', '<=', 20, 'PE Ratio <= 20'),
                    ScreenerCriteria('pe_ratio', '>', 0, 'PE Ratio > 0'),
                    ScreenerCriteria('return_on_assets', '>=', 5, 'Return on Assets >= 5%'),
                    ScreenerCriteria('revenue_growth', '>', 0, 'Revenue Growth > 0%'),
                    ScreenerCriteria('profit_margins', '>', 0, 'Profitable'),
                ],
                requires_historical_data=False,
                requires_yahoo_data=True,
                available=True
            ),
            # Growth at Reasonable Price
            'growth_at_reasonable_price': ScreenerDefinition(
                id='growth_at_reasonable_price',
                name='Growth at Reasonable Price',
                description='Revenue Growth >= 10% & PEG <= 1.5 & Profit Margins > 5%',
                category=ScreenerCategory.GROWTH,
                criteria=[
                    ScreenerCriteria('revenue_growth', '>=', 10, 'Revenue Growth >= 10%'),
                    ScreenerCriteria('peg_ratio', '<=', 1.5, 'PEG Ratio <= 1.5'),
                    ScreenerCriteria('peg_ratio', '>', 0.0, 'PEG Ratio > 0'),
                    ScreenerCriteria('profit_margins', '>', 5, 'Profit Margins > 5%'),
                ],
                requires_historical_data=False,
                requires_yahoo_data=True,
                available=True
            ),

            # Oversold in Uptrend
            'oversold_uptrend': ScreenerDefinition(
                id='oversold_uptrend',
                name='Oversold in Uptrend',
                description='RSI <= 35 & Price > SMA(200) & Price <= SMA(20)',
                category=ScreenerCategory.TECHNICAL,
                criteria=[
                    ScreenerCriteria('rsi', '<=', 35, 'RSI <= 35 (Oversold)'),
                    ScreenerCriteria('current_price', '>', FieldRef('sma_200'), 'Price > SMA(200) (Uptrend)'),
                    ScreenerCriteria('current_price', '<=', FieldRef('sma_20'), 'Price <= SMA(20) (Pullback)'),
                ],
                requires_historical_data=True,
                requires_yahoo_data=True,
                available=True
            ),

            # High Short Interest
            'high_short_interest': ScreenerDefinition(
                id='high_short_interest',
                name='High Short Interest',
                description='Short Float >= 10% (Technical criteria temporarily disabled due to numpy/pandas compatibility issue)',
                category=ScreenerCategory.MOMENTUM,
                criteria=[
                    ScreenerCriteria('short_percent_of_float', '>=', 10, 'High short interest'),
                    ScreenerCriteria('current_price', '>', FieldRef('sma_50'), 'Reclaiming SMA50'),
                    # ScreenerCriteria('volume_ratio', '>=', 1.5, 'Accumulation'),
                ],
                requires_historical_data=True,
                requires_yahoo_data=True,
                available=True
            ),

            # Momentum Pullback
            'momentum_pullback': ScreenerDefinition(
                id='momentum_pullback',
                name='Momentum Pullback',
                description='RS(6m) >= +10pp vs SPY & Close <= SMA20 & RSI 35–50',
                category=ScreenerCategory.MOMENTUM,
                criteria=[
                    ScreenerCriteria('rs_6m_vs_spy', '>=', 10, 'RS >= +10pp over 6m'),
                    ScreenerCriteria('current_price', '<=', FieldRef('sma_20'), 'Pullback to SMA20'),
                    ScreenerCriteria('rsi', '>=', 35, 'RSI >= 35'),
                    ScreenerCriteria('rsi', '<=', 50, 'RSI <= 50'),
                ],
                requires_historical_data=True,
                requires_yahoo_data=True,
                available=True
            ),

            # Golden Cross + First Pullback
            'golden_cross': ScreenerDefinition(
                id='golden_cross',
                name='Golden Cross + First Pullback',
                description='SMA(50) crossed above SMA(200) within 60d & Close within -2%..0% of SMA(50)',
                category=ScreenerCategory.TECHNICAL,
                criteria=[
                    ScreenerCriteria('gc_days_since', '<=', 60, 'SMA50>200 crossed within 60d'),
                    ScreenerCriteria('gc_days_since', '>=', 0, 'Cross exists'),
                    ScreenerCriteria('gc_within_sma50_frac', '>=', -0.02, 'Close not > SMA50 by more than 0%'),
                    ScreenerCriteria('gc_within_sma50_frac', '<=', 0.00, 'Close within pullback window'),
                ],
                requires_historical_data=True,
                requires_yahoo_data=True,
                available=True
            ),

            # Volatility Contraction Pattern
            'volatility_contraction': ScreenerDefinition(
                id='volatility_contraction',
                name='Volatility Contraction Pattern',
                description='BBWidth(20) <= 30th pct of 6m & 20d vol < 60d vol',
                category=ScreenerCategory.TECHNICAL,
                criteria=[
                    ScreenerCriteria('bb_width_20', '<=', FieldRef('bb_width_20_p30_6m'), 'Tight vs 6m history'),
                    ScreenerCriteria('vol20_lt_vol60', '==', True, 'Volume contraction'),
                ],
                requires_historical_data=True,
                requires_yahoo_data=True,
                available=True
            ),

            # Rule of 40
            'rule_of_40': ScreenerDefinition(
                id='rule_of_40',
                name='Rule of 40',
                description='Revenue Growth + Profit Margin >= 40%',
                category=ScreenerCategory.GROWTH,
                criteria=[
                    ScreenerCriteria('rule_of_40_score', '>=', 40, 'Revenue Growth + Profit Margin >= 40%'),
                ],
                requires_historical_data=False,
                requires_yahoo_data=True,
                available=True
            ),
        }

    def get_available_screeners(self) -> List[ScreenerDefinition]:
        """Get only available screeners."""
        return [s for s in self._screeners.values() if s.available]

    def get_available_operators(self) -> List[str]:
        """Get list of available operators."""
        return ['>=', '<=', '>', '<', '==', '!=', 'in', 'not_in']

    def get_available_fields(self) -> List[str]:
        """Get list of available field names."""
        return [
            'return_on_equity', 'return_on_assets', 'free_cashflow_yield',
            'peg_ratio', 'revenue_growth', 'profit_margins', 'pe_ratio',
            'short_percent_of_float', 'fifty_two_week_change', 'rsi',
            'institutional_ownership', 'current_price', 'rule_of_40_score',
            'sma_20', 'sma_50', 'sma_200', 'volume_ratio', 'bb_width_20',
            'bb_width_20_p30_6m', 'gc_days_since', 'gc_within_sma50_frac',
            'vol20_lt_vol60', 'rs_6m_vs_spy'
        ]

    def validate(self) -> None:
        """Validate screener configuration for errors."""
        available_fields = set(self.get_available_fields())
        available_operators = set(self.get_available_operators())

        for screener_id, screener_def in self._screeners.items():
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

    def eval_criterion(self, fields: Dict[str, Any], criteria: ScreenerCriteria) -> tuple[bool, str]:
        """Evaluate a single criterion."""
        if criteria.operator not in OP_FUNCS:
            return False, f"unknown operator {criteria.operator}"

        if criteria.field not in fields:
            return False, f"missing field {criteria.field}"

        lhs = fields[criteria.field]
        rhs = fields[criteria.value.name] if isinstance(criteria.value, FieldRef) else criteria.value

        # Handle None and NaN values gracefully (critical for finance accuracy)
        if lhs is None or not _is_finite_value(lhs):
            return False, f"field {criteria.field} is None or non-finite"

        if isinstance(criteria.value, FieldRef) and (rhs is None or not _is_finite_value(rhs)):
            return False, f"field {criteria.value.name} is None or non-finite"

        try:
            ok = OP_FUNCS[criteria.operator](lhs, rhs)
        except Exception as e:
            return False, f"error evaluating {criteria.field} {criteria.operator} {criteria.value}: {e}"

        return ok, (criteria.description or f"{criteria.field} {criteria.operator} {criteria.value}")

    def eval_screener(self, fields: Dict[str, Any], screener_def: ScreenerDefinition) -> dict:
        """Evaluate a screener against field data."""
        results = []
        all_ok = True
        for criteria in screener_def.criteria:
            ok, reason = self.eval_criterion(fields, criteria)
            results.append({"ok": bool(ok), "reason": reason})
            all_ok = all_ok and ok

        return {
            "screener_id": screener_def.id,
            "passed": bool(all_ok),
            "details": results
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert only available screeners to dictionary format for API responses."""
        return {
            'screeners': [
                {
                    'id': screener.id,
                    'name': screener.name,
                    'description': screener.description,
                    'category': screener.category.value,
                    'criteria': [
                        {
                            'field': criteria.field,
                            'operator': criteria.operator,
                            'value': {'fieldRef': criteria.value.name} if isinstance(criteria.value, FieldRef) else criteria.value,
                            'description': criteria.description
                        }
                        for criteria in screener.criteria
                    ],
                    'requires_historical_data': screener.requires_historical_data,
                    'requires_yahoo_data': screener.requires_yahoo_data,
                    'available': screener.available
                }
                for screener in self._screeners.values() if screener.available
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
