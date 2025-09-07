"""
Screener Utilities
=================
Helper functions for screener evaluation and calculation.
"""

from itertools import combinations
import logging
from typing import Dict, List, Optional, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from screener_config import get_screener_config, FieldRef, _is_finite_value

logger = logging.getLogger(__name__)

# Expected field names in portfolio data (for validation)
EXPECTED_FIELDS = {
    'return_on_equity', 'return_on_assets', 'free_cashflow_yield',
    'peg_ratio', 'revenue_growth', 'profit_margins', 'pe_ratio',
    'short_percent_of_float', 'fifty_two_week_change', 'rsi',
    'institutional_ownership', 'current_price', 'rule_of_40_score'
}


def calculate_screener_results(portfolio_data: List[Dict]) -> None:
    """
    Calculate screener results for all holdings in portfolio data.

    Args:
        portfolio_data: List of holding dictionaries to update with screener results
    """
    if not portfolio_data:
        logger.warning("No portfolio data provided for screener calculation")
        return

    try:
        screener_config = get_screener_config()
        available_screeners = screener_config.get_available_screeners()

        if not available_screeners:
            logger.warning("No available screeners found")
            for holding_data in portfolio_data:
                holding_data['passedScreeners'] = []
            return

        logger.debug(f"Evaluating {len(available_screeners)} screeners for {len(portfolio_data)} holdings")

        # Calculate technical indicators for first holding to get complete field set
        if portfolio_data:
            first_holding = portfolio_data[0]
            technical_data = calculate_technical_indicators(first_holding)
            if technical_data:
                first_holding.update(technical_data)

            # Validate field consistency after technical indicators are added
            validate_screener_fields(available_screeners, first_holding)

        for holding_data in portfolio_data:
            passed_screeners = []

            # Calculate technical indicators once per holding (if not already done for first holding)
            if holding_data is not portfolio_data[0]:  # Skip if already calculated above
                technical_data = calculate_technical_indicators(holding_data)
                if technical_data:
                    holding_data.update(technical_data)

            # Check each available screener
            for screener_def in available_screeners:
                if not screener_def.available:
                    continue

                # For screeners requiring historical data, check if we have the data
                if screener_def.requires_historical_data:
                    # Check if we have the required technical fields
                    required_fields = set()
                    for criteria in screener_def.criteria:
                        if isinstance(criteria.value, FieldRef):
                            required_fields.add(criteria.value.name)
                        required_fields.add(criteria.field)

                    # Skip if we don't have the required technical data (check for None and NaN)
                    if not all(field in holding_data and holding_data[field] is not None and _is_finite_value(holding_data[field]) for field in required_fields):
                        continue

                # Check if holding passes this screener's criteria using new evaluation engine
                result = screener_config.eval_screener(holding_data, screener_def)
                if result['passed']:
                    passed_screeners.append(screener_def.id)
                    holding_data['screener_score'] += screener_def.weight

            holding_data['passedScreeners'] = passed_screeners
            # Bonus for combinations of screeners
            screener_pairs = {
                tuple(sorted((a, b))) for a, b in combinations(set(passed_screeners), 2)
                if (b in screener_config.screeners[a].combine_with
                    or a in screener_config.screeners[b].combine_with)
                   and screener_config.screeners[a].category != screener_config.screeners[b].category  # optional
            }
            holding_data['screener_score'] += min(5, 2 * len(screener_pairs))

        # Log summary for debugging
        total_matches = sum(len(h.get('passedScreeners', [])) for h in portfolio_data)
        logger.debug(f"Screener evaluation complete: {total_matches} total matches across {len(portfolio_data)} holdings")

    except ImportError as e:
        logger.error(f"Failed to import screener configuration: {e}")
        for holding_data in portfolio_data:
            holding_data['passedScreeners'] = []
    except Exception as e:
        logger.error(f"Unexpected error in screener calculation: {e}")
        # Continue without screener results if calculation fails
        for holding_data in portfolio_data:
            holding_data['passedScreeners'] = []


def validate_screener_fields(available_screeners: List, sample_holding: Dict) -> None:
    """
    Validate that screener field names match portfolio data fields.

    Args:
        available_screeners: List of available screener definitions
        sample_holding: Sample holding data to check field availability
    """
    if not sample_holding:
        return

    used_fields = set()
    for screener_def in available_screeners:
        for criteria in screener_def.criteria:
            used_fields.add(criteria.field)
            # Also check FieldRef names
            if isinstance(criteria.value, FieldRef):
                used_fields.add(criteria.value.name)

    missing_fields = used_fields - set(sample_holding.keys())
    if missing_fields:
        logger.error(f"CRITICAL: Screener fields not found in portfolio data: {missing_fields}")
        logger.error("This will cause incorrect screener results!")

    unexpected_fields = set(sample_holding.keys()) - EXPECTED_FIELDS - {'name', 'symbol', 'quantity', 'current_price', 'market_value', 'profit', 'return_pct', 'portfolio_pct', 'date', 'passedScreeners'}
    if unexpected_fields:
        logger.debug(f"Unexpected fields in portfolio data: {unexpected_fields}")


def calculate_technical_indicators(holding_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Calculate technical indicators for a holding using price history data.

    Args:
        holding_data: Dictionary containing holding data including price history

    Returns:
        Dictionary with technical indicators or None if calculation fails
    """
    try:
        # Get current price and RSI from holding data
        current_price = holding_data.get('current_price')
        rsi = holding_data.get('rsi')

        if current_price is None:
            logger.debug(f"No current price available for {holding_data.get('name', 'unknown')}")
            return None

        # Technical indicators are now calculated in main.py from price history data
        # This function is no longer needed for technical indicators calculation
        # Return None to indicate no additional technical indicators to calculate
        return None

    except Exception as e:
        logger.error(f"Failed to calculate technical indicators: {e}")
        return None
