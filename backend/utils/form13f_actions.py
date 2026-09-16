"""Quantity adjustments for 13F comparisons, never price-return adjustments.

Yahoo's split feed mixes share splits and spin-off price adjustments. Explicit
exceptions below are sourced in docs/reviews/2026-09-16-first-batch-fable-review.md.
Unrecognised ratios produce an unknown comparison, not an invented trade.
"""

from math import isclose, isfinite

# Dates are the feed's ex-dates. Overrides also cover missing feed entries.
QUANTITY_OVERRIDES = {
    "FDX": {"2026-06-01": 1.0},
    "HON": {"2026-06-29": 0.5},
    "SCCO": {"2026-02-10": 1.0085, "2026-05-13": 1.01, "2026-08-11": 1.012},
}

# Same security after the reverse split; do not conflate the separate HONA spin-off.
CUSIP_ALIASES = {"438516106": "438516205"}
HON_CUSIPS = ("438516106", "438516205")


def canonical_cusip(cusip: str) -> str:
    cusip = cusip.upper()
    return CUSIP_ALIASES.get(cusip, cusip)


def quantity_split_history(splits: dict | None, symbol: str | None = None) -> dict:
    """Keep common split ratios; flag unusual feed values unless verified explicitly.

    This is a conservative fallback, not a corporate-action classifier. Verified
    exceptions take precedence even when their feed value looks like a normal split.
    """
    result = {}
    for stamp, factor in (splits or {}).items():
        key = str(stamp)[:10]
        valid = isinstance(factor, (int, float)) and not isinstance(factor, bool) and isfinite(factor) and factor > 0
        common = valid and (
            isclose(factor, round(factor), rel_tol=0, abs_tol=1e-8)
            or isclose(1 / factor, round(1 / factor), rel_tol=0, abs_tol=1e-8)
            or any(isclose(factor, r, rel_tol=0, abs_tol=1e-8) for r in (1.5, 2 / 3, 1.25, 0.8, 4 / 3, 0.75))
        )
        result[key] = factor if common else None
    result.update(QUANTITY_OVERRIDES.get(symbol, {}))
    return result
