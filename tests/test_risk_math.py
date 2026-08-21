"""Known-answer tests for the portfolio-risk primitives.

Focused on `effective_bets`: the metric it replaces (inverse Herfindahl of risk
contributions) passed every plausible smoke test while being blind to
correlation, so the cases here are chosen to separate the two.

Needs numpy/scipy — unlike tests/test_agent_constraints.py this cannot run on
bare stdlib.
"""

import math

import numpy as np
import pytest

from backend.utils.risk_math import effective_bets, ledoit_wolf_cov, risk_contributions

N = 8


def _equal_weights(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def test_uncorrelated_equal_weight_gives_n_bets():
    cov = np.eye(N) * 0.04
    assert effective_bets(cov, _equal_weights(N)) == pytest.approx(N)


def test_perfectly_correlated_gives_one_bet():
    cov = np.full((N, N), 0.04)
    assert effective_bets(cov, _equal_weights(N)) == pytest.approx(1.0)


def test_concentrated_weights_give_one_bet_despite_independence():
    """The case that separates this from a universe-level measure.

    Eight independent assets, 99% of capital in one: the *opportunity set* is
    still eight bets, but the *portfolio* is one.
    """
    cov = np.eye(N) * 0.04
    w = np.full(N, 0.01 / (N - 1))
    w[0] = 0.99
    assert effective_bets(cov, w) == pytest.approx(1.0, abs=0.05)


def test_two_clone_pairs_count_as_two_bets():
    """Four positions, two identical pairs — the inverse-Herfindahl measure
    reports ~4 here, which is the bug this metric exists to avoid."""
    corr = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    cov = corr * 0.04
    w = _equal_weights(4)
    assert effective_bets(cov, w) == pytest.approx(2.0)

    rc, _ = risk_contributions(cov, w)
    herfindahl_bets = 1.0 / (rc**2).sum()
    assert herfindahl_bets == pytest.approx(4.0)


def test_bounded_by_position_count_on_a_random_book():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.02, size=(600, 12))
    cov, _ = ledoit_wolf_cov(returns)
    bets = effective_bets(cov, _equal_weights(12))
    assert 1.0 <= bets <= 12.0


def test_singular_covariance_does_not_nan():
    """A duplicated column makes the sample covariance rank-deficient; eigh can
    then return small negative eigenvalues."""
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0, 0.02, size=(300, 5))
    returns = np.hstack([returns, returns[:, :1]])
    cov = np.cov(returns.T)
    bets = effective_bets(cov, _equal_weights(6))
    assert math.isfinite(bets)
    assert bets >= 1.0
