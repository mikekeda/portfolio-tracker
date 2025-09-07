"""
Portfolio Utilities
==================
Helper functions for portfolio calculations and data processing.
"""

from typing import Dict


def weighted_add(target: Dict[str, float], weights: Dict[str, float], value: float) -> None:
    """
    Add value into target buckets using percentage weights.

    Args:
        target: Dictionary to add values to
        weights: Dictionary of percentage weights
        value: Value to distribute
    """
    for bucket, pct in weights.items():
        target[bucket] = target.get(bucket, 0.0) + value * pct / 100.0
