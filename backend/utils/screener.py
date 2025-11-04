"""
Screener Utilities
=================
Helper functions for screener evaluation and calculation.
"""

from itertools import combinations

from backend.screener_config import FieldRef, get_screener_config
from config import logger

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


def calculate_screener_results(portfolio_data: list[dict]) -> None:
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
                holding_data["screener_score"] = 0
            return

        logger.debug(f"Evaluating {len(available_screeners)} screeners for {len(portfolio_data)} holdings")

        for holding_data in portfolio_data:
            passed_screeners = []
            # Initialize screener_score if it doesn't exist
            holding_data["screener_score"] = holding_data.get("screener_score", 0)

            # Check each available screener
            for screener_def in available_screeners:
                if not screener_def.available:
                    continue

                # Check if holding passes this screener's criteria using the evaluation engine
                result = screener_config.eval_screener(holding_data, screener_def)
                if result["passed"]:
                    passed_screeners.append(screener_def.id)
                    holding_data["screener_score"] += screener_def.weight

            holding_data["passedScreeners"] = passed_screeners

            # --- SUGGESTION: Refined Combination Bonus Logic ---
            # This bonus rewards stocks that pass screeners from different, complementary categories.
            # For example, a stock with strong fundamentals (Quality) that is also in a technical
            # uptrend (Momentum) is a more robust candidate.
            screener_pairs = {
                tuple(sorted((a, b)))
                for a, b in combinations(set(passed_screeners), 2)
                if (
                    (b in screener_config.screeners[a].combine_with or a in screener_config.screeners[b].combine_with)
                    and screener_config.screeners[a].category != screener_config.screeners[b].category
                )
            }

            # Increased the cap from 5 to 10 to give more weight to exceptional stocks
            # that pass multiple combinations.
            combination_bonus = 2 * len(screener_pairs)
            holding_data["screener_score"] += min(10, combination_bonus)

        # Log summary for debugging
        total_matches = sum(len(h.get("passedScreeners", [])) for h in portfolio_data)
        logger.debug(
            f"Screener evaluation complete: {total_matches} total matches across {len(portfolio_data)} holdings"
        )

    except ImportError as e:
        logger.error(f"Failed to import screener configuration: {e}")
        for holding_data in portfolio_data:
            holding_data["passedScreeners"] = []
            holding_data["screener_score"] = 0
    except Exception as e:
        logger.error(f"Unexpected error in screener calculation: {e}", exc_info=True)
        # Continue without screener results if calculation fails
        for holding_data in portfolio_data:
            holding_data["passedScreeners"] = []
            holding_data["screener_score"] = 0


def validate_screener_fields(available_screeners: list, sample_holding: dict) -> None:
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
        # This is downgraded to a warning, as some data might not be available for all stocks,
        # which is a normal occurrence. The eval engine handles this gracefully.
        logger.warning(f"Screener fields not found in sample holding data: {missing_fields}")

    # This validation can be noisy if technical indicators haven't been added yet.
    # It's more for initial setup debugging.
    all_known_fields = (
        EXPECTED_FIELDS
        | used_fields
        | {
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
            "screener_score",
        }
    )

    unexpected_fields = set(sample_holding.keys()) - all_known_fields

    if unexpected_fields:
        logger.debug(f"Fields in holding data not defined in screeners: {unexpected_fields}")
