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

        logger.debug(
            "Evaluating %s screeners for %s holdings",
            len(available_screeners),
            len(portfolio_data),
        )

        # Local references shave attribute lookups in the hot loop (important when
        # "Show all" mode feeds ~700 monitored stocks through here).
        passes_screener = screener_config.passes_screener
        screeners_map = screener_config.screeners

        for holding_data in portfolio_data:
            passed_screeners: list[str] = []
            score = 0

            for screener_def in available_screeners:
                if passes_screener(holding_data, screener_def):
                    passed_screeners.append(screener_def.id)
                    score += screener_def.weight

            holding_data["passedScreeners"] = passed_screeners

            # Combination bonus: reward stocks that pass mutually-endorsed screener pairs.
            # Complementarity is fully delegated to `combine_with` (curated whitelist), so we
            # no longer gate by category — two VALUE screeners endorsing each other are a
            # legitimate double-confirmation. Diminishing returns (pairs ** 0.85) prevent a
            # single stock with many passes from saturating the score while still giving
            # exceptional stocks room to stand out.
            if len(passed_screeners) >= 2:
                pair_count = 0
                for a, b in combinations(passed_screeners, 2):
                    if b in screeners_map[a].combine_with_set or a in screeners_map[b].combine_with_set:
                        pair_count += 1
                if pair_count:
                    score += round(2 * pair_count**0.85)

            holding_data["screener_score"] = score

        # Log summary for debugging
        total_matches = sum(len(h.get("passedScreeners", [])) for h in portfolio_data)
        logger.debug(
            "Screener evaluation complete: %s total matches across %s holdings",
            total_matches,
            len(portfolio_data),
        )

    except ImportError as e:
        logger.error("Failed to import screener configuration: %s", e)
        for holding_data in portfolio_data:
            holding_data["passedScreeners"] = []
            holding_data["screener_score"] = 0
    except Exception as e:
        logger.error("Unexpected error in screener calculation: %s", e, exc_info=True)
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
        logger.warning("Screener fields not found in sample holding data: %s", missing_fields)

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
        logger.debug("Fields in holding data not defined in screeners: %s", unexpected_fields)
