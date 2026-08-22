"""
ETF Look-Through Aggregates
===========================
Fund-level fundamentals derived from the constituents in `etf_holdings`.

Each metric needs a specific aggregation and the choice is not cosmetic: a
weighted *arithmetic* P/E overstates these funds by 37-80% because one 339x
constituent on a 2% weight dominates the mean, while the harmonic form
(1 / sum(w/x), the reciprocal of cap-weighted earnings yield) reproduces the
published index multiple. METRIC_AGGREGATION is the whole taxonomy — adding a
metric is one row, not a new branch.

Pure stdlib so the arithmetic is testable without a database.
"""

import math
from dataclasses import dataclass

# Below this share of the fund's published weight the mean describes the
# constituents we happen to track, not the fund. R2SC.L resolves at 7%.
MIN_COVERAGE = 0.80

HARMONIC = "harmonic"
MEAN = "mean"

# What each weighted mean is weighted *by*. Published weights are market-cap
# shares, which is right for anything priced off market cap and wrong for
# anything measured against revenue: cap-weighting a margin over-weights
# high-multiple, low-revenue names, reading VUAG's net margin at 26.8% against
# a published S&P figure near 12-13%. Revenue-weighting is the exact
# sum(numerator)/sum(denominator) form and reads 13.9%.
CAP_WEIGHT = "cap"
REVENUE_WEIGHT = "revenue"

# metric key on the holding dict -> (aggregation, weight basis, is_approximate)
#
# Approximate marks ratios over a denominator we cannot reconstruct, so the
# weighted mean of the ratio is the best available and must be labelled in the
# UI rather than presented as the fund's true figure.
METRIC_AGGREGATION: dict[str, tuple[str, str, bool]] = {
    # Price multiples: price over a flow, so the fund's multiple is
    # sum(w*P)/sum(w*E). Non-positive values are dropped, not netted.
    "pe_ratio": (HARMONIC, CAP_WEIGHT, False),
    "forward_pe_ratio": (HARMONIC, CAP_WEIGHT, False),
    "ps_ratio": (HARMONIC, CAP_WEIGHT, False),
    # Also harmonic and exact: PEG = PE/g, so sum(w)/sum(w/PEG) is the fund's PE
    # over its earnings-weighted growth — what a fund-level PEG means.
    "peg_ratio": (HARMONIC, CAP_WEIGHT, False),
    # Measured against revenue, so revenue-weighting makes these exact.
    "profit_margins": (MEAN, REVENUE_WEIGHT, False),
    "gross_margin": (MEAN, REVENUE_WEIGHT, False),
    "operating_margin": (MEAN, REVENUE_WEIGHT, False),
    # Exact aggregate weights by *prior*-period revenue; current revenue
    # over-weights the fastest growers slightly. Far closer than cap weighting.
    "revenue_growth": (MEAN, REVENUE_WEIGHT, True),
    # Denominators we cannot reconstruct: invested capital, equity, assets.
    "roic": (MEAN, CAP_WEIGHT, True),
    "return_on_equity": (MEAN, CAP_WEIGHT, True),
    "return_on_assets": (MEAN, CAP_WEIGHT, True),
    "debtToEquity": (MEAN, CAP_WEIGHT, True),
    "quickRatio": (MEAN, CAP_WEIGHT, True),
    # Bounded or already normalised, so the mean is meaningful on its own scale.
    "f_score": (MEAN, CAP_WEIGHT, False),
    "recommendation_mean": (MEAN, CAP_WEIGHT, False),
    "institutional_ownership": (MEAN, CAP_WEIGHT, False),
    "form13f_score": (MEAN, CAP_WEIGHT, False),
    # Exact under cap weighting: the denominator *is* market cap, so
    # sum(w * FCF/MC) = sum(FCF)/sum(MC) identically.
    "free_cashflow_yield": (MEAN, CAP_WEIGHT, False),
    # Analyst target upside is a return, so cap weighting is exact: it is what
    # the fund returns if every constituent reaches its median target.
    "prediction": (MEAN, CAP_WEIGHT, False),
    # Computed by the caller as screener_score / screener_score_max. Aggregated
    # as the mean of the ratios: score_max is sector-dependent (23-60), so
    # aggregating numerator and denominator separately is a different quantity.
    "screener_ratio": (MEAN, CAP_WEIGHT, False),
}

# Deliberately absent: market_cap. The weighted average of constituent market
# caps is "typical company size in this fund", not the fund's own size, and the
# Mkt Cap column renders raw quote-currency values without converting — so the
# figure would be both a different quantity and in an ambiguous unit.


# Metrics whose value is an approximation of the exact sum/sum form.
APPROXIMATE_METRICS = frozenset(m for m, (_, _, approx) in METRIC_AGGREGATION.items() if approx)


@dataclass(frozen=True)
class Aggregate:
    """An aggregated metric with the share of fund weight standing behind it."""

    value: float
    coverage: float
    n: int


def _usable(pairs: list[tuple[float, float | None]], positive_only: bool) -> list[tuple[float, float]]:
    """Drop pairs that cannot contribute: missing, non-finite, or non-positive weight."""
    kept = []
    for weight, value in pairs:
        if weight is None or value is None:
            continue
        if not isinstance(weight, (int, float)) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(weight) or not math.isfinite(value) or weight <= 0:
            continue
        if positive_only and value <= 0:
            continue
        kept.append((float(weight), float(value)))
    return kept


def weighted_harmonic(pairs: list[tuple[float, float | None]], total_weight: float) -> Aggregate | None:
    """Cap-weighted harmonic mean of a price multiple; None when nothing is usable.

    Non-positive multiples are excluded rather than netted — a loss-maker has no
    meaningful P/E, and a negative one would cancel a positive one.
    """
    kept = _usable(pairs, positive_only=True)
    if not kept or total_weight <= 0:
        return None

    covered = sum(w for w, _ in kept)
    # Renormalising over the covered set is what makes this a multiple rather
    # than a figure scaled by how much of the fund happened to be missing.
    reciprocal = sum(w / v for w, v in kept) / covered
    if reciprocal <= 0:
        return None
    return Aggregate(1.0 / reciprocal, covered / total_weight, len(kept))


def weighted_mean(pairs: list[tuple[float, float | None]], total_weight: float) -> Aggregate | None:
    """Cap-weighted arithmetic mean; None when nothing is usable."""
    kept = _usable(pairs, positive_only=False)
    if not kept or total_weight <= 0:
        return None

    covered = sum(w for w, _ in kept)
    return Aggregate(sum(w * v for w, v in kept) / covered, covered / total_weight, len(kept))


def revenue_weight(cap_weight: float, ps_ratio: float | None) -> float | None:
    """Revenue share implied by a cap weight, since P/S is market cap over revenue.

    None when P/S is missing or non-positive: without it the constituent's
    revenue is unknown, and guessing would silently reweight the whole fund.
    """
    if not ps_ratio or ps_ratio <= 0 or not math.isfinite(ps_ratio):
        return None
    return cap_weight / ps_ratio


def revenue_weighted_mean(
    triples: list[tuple[float, float | None, float | None]], total_weight: float
) -> Aggregate | None:
    """Revenue-weighted mean, with coverage measured in the fund's own cap weights.

    `triples` are (cap weight, revenue weight, value). The value is weighted by
    revenue because that is the exact sum(numerator)/sum(denominator) form, but
    coverage must stay a share of the *fund* — revenue weights only exist for
    constituents we resolved, so a share of them would read ~100% for a fund we
    barely track.
    """
    usable = [
        (cap, rev, value)
        for cap, rev, value in triples
        if rev is not None and value is not None and isinstance(value, (int, float)) and math.isfinite(value)
    ]
    if not usable or total_weight <= 0:
        return None

    revenue_total = sum(rev for _, rev, _ in usable)
    if revenue_total <= 0:
        return None

    value = sum(rev * v for _, rev, v in usable) / revenue_total
    return Aggregate(value, sum(cap for cap, _, _ in usable) / total_weight, len(usable))


def aggregate_fund(
    constituents: list[tuple[float, dict]],
    total_weight: float,
    min_coverage: float = MIN_COVERAGE,
) -> tuple[dict[str, float], dict[str, float]]:
    """Aggregate every metric in the taxonomy over one fund's constituents.

    `constituents` pairs each resolved constituent's published cap weight with
    its metric dict; `total_weight` is the fund's full published weight including
    constituents that are not tracked, so coverage reflects the fund rather than
    the sample.

    Returns (values, coverage_pct). Coverage is reported for every metric that
    produced a figure; values holds only those at or above `min_coverage`.
    """
    revenue_weights = [revenue_weight(weight, metrics.get("ps_ratio")) for weight, metrics in constituents]

    values: dict[str, float] = {}
    coverage: dict[str, float] = {}

    for metric, (kind, basis, _approximate) in METRIC_AGGREGATION.items():
        if basis == REVENUE_WEIGHT:
            triples = [
                (weight, rev, metrics.get(metric))
                for rev, (weight, metrics) in zip(revenue_weights, constituents)
            ]
            aggregate = revenue_weighted_mean(triples, total_weight)
        else:
            pairs = [(weight, metrics.get(metric)) for weight, metrics in constituents]
            aggregate = (
                weighted_harmonic(pairs, total_weight) if kind == HARMONIC else weighted_mean(pairs, total_weight)
            )
        if aggregate is None:
            continue

        coverage[metric] = round(aggregate.coverage * 100.0, 1)
        if aggregate.coverage >= min_coverage:
            values[metric] = aggregate.value

    return values, coverage
