"""
Screener Utilities
=================
Helper functions for screener evaluation and calculation.
"""

import logging
from itertools import combinations
from typing import Dict, List

from backend.screener_config import FieldRef, _is_finite_value, get_screener_config

logger = logging.getLogger(__name__)

# Expected field names in portfolio data (for validation)
EXPECTED_FIELDS = {
    "return_on_equity",
    "return_on_assets",
    "free_cashflow_yield",
    "peg_ratio",
    "revenue_growth",
    "profit_margins",
    "pe_ratio",
    "short_percent_of_float",
    "fifty_two_week_change",
    "rsi",
    "institutional_ownership",
    "current_price",
    "rule_of_40_score",
}


def calculate_screener_results(portfolio_data: List[Dict]) -> None:
    """Calculate screener results for all holdings in portfolio data."""
    if not portfolio_data:
        logger.warning("No portfolio data provided for screener calculation")
        return

    try:
        screener_config = get_screener_config()
        available_screeners = screener_config.get_available_screeners()

        if not available_screeners:
            logger.warning("No available screeners found")
            for holding_data in portfolio_data:
                holding_data["passedScreeners"] = []
            return

        logger.debug(f"Evaluating {len(available_screeners)} screeners for {len(portfolio_data)} holdings")

        # Calculate technical indicators for first holding to get complete field set
        if portfolio_data:
            first_holding = portfolio_data[0]

            # Validate field consistency after technical indicators are added
            validate_screener_fields(available_screeners, first_holding)

        for holding_data in portfolio_data:
            passed_screeners = []

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
                    if not all(
                        field in holding_data
                        and holding_data[field] is not None
                        and _is_finite_value(holding_data[field])
                        for field in required_fields
                    ):
                        continue

                # Check if holding passes this screener's criteria using new evaluation engine
                result = screener_config.eval_screener(holding_data, screener_def)
                if result["passed"]:
                    passed_screeners.append(screener_def.id)
                    holding_data["screener_score"] += screener_def.weight

            holding_data["passedScreeners"] = passed_screeners
            # Bonus for combinations of screeners
            screener_pairs = {
                tuple(sorted((a, b)))
                for a, b in combinations(set(passed_screeners), 2)
                if (
                    (b in screener_config.screeners[a].combine_with or a in screener_config.screeners[b].combine_with)
                    and screener_config.screeners[a].category != screener_config.screeners[b].category  # optional
                )
            }
            holding_data["screener_score"] += min(5, 2 * len(screener_pairs))

        # Log summary for debugging
        total_matches = sum(len(h.get("passedScreeners", [])) for h in portfolio_data)
        logger.debug(
            f"Screener evaluation complete: {total_matches} total matches across {len(portfolio_data)} holdings"
        )

    except ImportError as e:
        logger.error(f"Failed to import screener configuration: {e}")
        for holding_data in portfolio_data:
            holding_data["passedScreeners"] = []
    except Exception as e:
        logger.error(f"Unexpected error in screener calculation: {e}")
        # Continue without screener results if calculation fails
        for holding_data in portfolio_data:
            holding_data["passedScreeners"] = []


def validate_screener_fields(available_screeners: List, sample_holding: Dict) -> None:
    """Validate that screener field names match portfolio data fields."""
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

    unexpected_fields = (
        set(sample_holding.keys())
        - EXPECTED_FIELDS
        - {
            "name",
            "symbol",
            "quantity",
            "current_price",
            "market_value",
            "profit",
            "return_pct",
            "portfolio_pct",
            "date",
            "passedScreeners",
        }
    )
    if unexpected_fields:
        logger.debug(f"Unexpected fields in portfolio data: {unexpected_fields}")
