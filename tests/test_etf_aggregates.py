"""Arithmetic of the ETF look-through aggregates.

The failure mode here is a plausible-looking wrong number: an arithmetic mean
where a harmonic one belongs, a renormalisation that was dropped, a negative
multiple netted against a positive one. Every case below separates the correct
implementation from one that still returns a number.

Pure stdlib:

    python tests/test_etf_aggregates.py
    pytest tests/test_etf_aggregates.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.utils.etf_aggregates import (
    APPROXIMATE_METRICS,
    CAP_WEIGHT,
    HARMONIC,
    MEAN,
    METRIC_AGGREGATION,
    MIN_COVERAGE,
    REVENUE_WEIGHT,
    aggregate_fund,
    revenue_weight,
    weighted_harmonic,
    weighted_mean,
)

TOLERANCE = 1e-9


def test_harmonic_is_not_the_arithmetic_mean():
    # Two equal-weighted P/Es of 10 and 30 buy 1/10 + 1/30 of earnings per unit
    # of price, so the fund trades at 15x — not the 20x an arithmetic mean gives.
    assert abs(weighted_harmonic([(0.5, 10.0), (0.5, 30.0)], 1.0).value - 15.0) < TOLERANCE
    assert abs(weighted_mean([(0.5, 10.0), (0.5, 30.0)], 1.0).value - 20.0) < TOLERANCE


def test_harmonic_renormalises_weights_that_do_not_sum_to_one():
    # Weights arrive as published percentages and never sum to 1 once unresolved
    # constituents are dropped. Without renormalising, the result scales with
    # however much of the fund happened to be missing.
    as_fractions = weighted_harmonic([(0.5, 10.0), (0.5, 30.0)], 1.0)
    as_percentages = weighted_harmonic([(50.0, 10.0), (50.0, 30.0)], 100.0)
    partial = weighted_harmonic([(5.0, 10.0), (5.0, 30.0)], 100.0)

    assert abs(as_percentages.value - 15.0) < TOLERANCE
    assert abs(partial.value - 15.0) < TOLERANCE, "value must not depend on covered weight"
    assert abs(as_fractions.value - as_percentages.value) < TOLERANCE
    assert abs(partial.coverage - 0.10) < TOLERANCE, "coverage must depend on it"


def test_mean_renormalises_too():
    assert abs(weighted_mean([(5.0, 10.0), (5.0, 30.0)], 100.0).value - 20.0) < TOLERANCE


def test_non_positive_multiples_are_excluded_not_netted():
    # A loss-maker has no meaningful P/E. Netting -20 against +20 would cancel
    # the pair to nothing; excluding leaves the profitable half at its own value.
    aggregate = weighted_harmonic([(0.5, 20.0), (0.5, -20.0)], 1.0)
    assert abs(aggregate.value - 20.0) < TOLERANCE
    assert aggregate.n == 1
    assert abs(aggregate.coverage - 0.5) < TOLERANCE


def test_negative_values_are_kept_by_the_arithmetic_mean():
    # Unlike a multiple, a negative margin or growth rate is a real observation.
    assert abs(weighted_mean([(0.5, 10.0), (0.5, -30.0)], 1.0).value - (-10.0)) < TOLERANCE


def test_missing_and_non_finite_inputs_are_dropped():
    assert abs(weighted_mean([(0.5, 10.0), (0.5, None)], 1.0).value - 10.0) < TOLERANCE
    assert abs(weighted_mean([(0.5, 10.0), (0.5, float("nan"))], 1.0).value - 10.0) < TOLERANCE
    assert weighted_mean([(0.5, None), (0.5, None)], 1.0) is None
    assert weighted_harmonic([], 1.0) is None
    assert weighted_mean([(0.5, 10.0)], 0.0) is None, "no fund weight means no coverage denominator"


def test_coverage_gate_drops_the_value_but_keeps_the_coverage():
    # R2SC.L resolves at 7%: the figure must not be published, but the integrity
    # check needs to see why it was withheld.
    thin = [(7.0, {"pe_ratio": 20.0, "roic": 15.0})]
    values, coverage = aggregate_fund(thin, total_weight=100.0)

    assert values == {}
    assert abs(coverage["pe_ratio"] - 7.0) < TOLERANCE
    assert abs(coverage["roic"] - 7.0) < TOLERANCE


def test_coverage_gate_admits_a_well_covered_fund():
    constituents = [(60.0, {"pe_ratio": 10.0}), (30.0, {"pe_ratio": 30.0})]
    values, coverage = aggregate_fund(constituents, total_weight=100.0)

    assert abs(coverage["pe_ratio"] - 90.0) < TOLERANCE
    assert abs(values["pe_ratio"] - 1.0 / ((60.0 / 10.0 + 30.0 / 30.0) / 90.0)) < TOLERANCE


def test_gate_is_applied_per_metric_not_per_fund():
    # A fund can carry P/E for everything and ROIC for a third of the book.
    constituents = [
        (60.0, {"pe_ratio": 20.0, "roic": 15.0}),
        (40.0, {"pe_ratio": 20.0}),
    ]
    values, coverage = aggregate_fund(constituents, total_weight=100.0)

    assert "pe_ratio" in values
    assert "roic" not in values
    assert abs(coverage["roic"] - 60.0) < TOLERANCE


def test_margins_are_revenue_weighted_not_cap_weighted():
    # A high-multiple constituent carries a large cap weight and a small share of
    # revenue. Cap-weighting its margin drags the fund's figure toward it; the
    # exact aggregate is sum(profit)/sum(revenue), which is revenue-weighted.
    # 90% of cap at P/S 30 and margin 40%, 10% of cap at P/S 1 and margin 5%:
    # revenue shares are 3 and 10, so the fund's margin is nearer the 5% name.
    constituents = [
        (90.0, {"ps_ratio": 30.0, "profit_margins": 40.0}),
        (10.0, {"ps_ratio": 1.0, "profit_margins": 5.0}),
    ]
    values, _ = aggregate_fund(constituents, total_weight=100.0)

    expected = (3.0 * 40.0 + 10.0 * 5.0) / 13.0
    assert abs(values["profit_margins"] - expected) < TOLERANCE
    assert values["profit_margins"] < 15.0, "cap weighting would give 36.5%"


def test_revenue_weight_needs_a_usable_ps_ratio():
    assert abs(revenue_weight(10.0, 2.0) - 5.0) < TOLERANCE
    assert revenue_weight(10.0, None) is None
    assert revenue_weight(10.0, 0.0) is None
    assert revenue_weight(10.0, -2.0) is None, "a negative P/S would flip the weight's sign"


def test_a_constituent_without_ps_lowers_revenue_weighted_coverage():
    # It contributes no revenue share, so it is absent from the aggregate — the
    # coverage figure has to say so rather than reporting the fund as complete.
    constituents = [
        (70.0, {"ps_ratio": 2.0, "profit_margins": 20.0}),
        (30.0, {"profit_margins": 10.0}),
    ]
    values, coverage = aggregate_fund(constituents, total_weight=100.0)

    assert abs(coverage["profit_margins"] - 70.0) < TOLERANCE
    assert "profit_margins" not in values, "70% is below the gate"


def test_taxonomy_is_well_formed():
    for metric, (kind, basis, approximate) in METRIC_AGGREGATION.items():
        assert kind in (HARMONIC, MEAN), f"{metric}: {kind}"
        assert basis in (CAP_WEIGHT, REVENUE_WEIGHT), f"{metric}: {basis}"
        assert not (kind == HARMONIC and basis == REVENUE_WEIGHT), f"{metric}: multiples price off cap"
        assert isinstance(approximate, bool), metric
    for metric in ("profit_margins", "gross_margin", "operating_margin", "revenue_growth"):
        assert METRIC_AGGREGATION[metric][1] == REVENUE_WEIGHT, metric
    # Multiples must never be averaged arithmetically.
    for metric in ("pe_ratio", "forward_pe_ratio", "ps_ratio"):
        assert METRIC_AGGREGATION[metric][0] == HARMONIC
    # A ratio whose denominator is market cap aggregates exactly under cap
    # weighting, so it must not be flagged approximate.
    assert "free_cashflow_yield" not in APPROXIMATE_METRICS
    assert "screener_ratio" not in APPROXIMATE_METRICS
    assert "roic" in APPROXIMATE_METRICS


def test_min_coverage_matches_the_documented_gate():
    assert MIN_COVERAGE == 0.80


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
