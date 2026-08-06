"""Cumulative split-factor helpers for aligning as-reported share counts with adjusted prices."""

from __future__ import annotations

from datetime import date, datetime
from math import prod
from typing import Any, Mapping


def _parse_split_date(key: str | date | datetime) -> date | None:
    """Parse a splits JSONB key to a date."""
    if isinstance(key, date) and not isinstance(key, datetime):
        return key
    if isinstance(key, datetime):
        return key.date()
    text = str(key).strip()
    if not text:
        return None
    # Yahoo scrub_for_json may leave "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS"
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def cumulative_split_factor(
    splits: Mapping[str, Any] | None,
    as_of: date,
    *,
    after: bool = True,
) -> float:
    """Product of split factors with date > `as_of` (default) or >= `as_of`.

    Uses the full split history deliberately: prices are retroactively adjusted
    by the same future splits, so market cap stays invariant. Truncated and full
    backtests must both see the complete `InstrumentYahoo.splits` blob — never
    "splits known as of as_of".

    Factors are floats (GOOGL 1.998) and may be < 1 (reverse splits).
    """
    if not splits:
        return 1.0

    factors: list[float] = []
    for raw_key, raw_factor in splits.items():
        split_date = _parse_split_date(raw_key)
        if split_date is None:
            continue
        try:
            factor = float(raw_factor)
        except (TypeError, ValueError):
            continue
        if factor == 0.0:
            continue
        if after and split_date > as_of:
            factors.append(factor)
        elif not after and split_date >= as_of:
            factors.append(factor)

    return float(prod(factors)) if factors else 1.0


def adjust_share_count(
    shares: float | None,
    splits: Mapping[str, Any] | None,
    count_ref_date: date,
) -> float | None:
    """Put an as-reported share count on today's split-adjusted basis."""
    if shares is None:
        return None
    return float(shares) * cumulative_split_factor(splits, count_ref_date)
