"""
Pure portfolio-risk math: Ledoit-Wolf shrunk covariance, Hierarchical Risk
Parity (HRP) weights, and risk-contribution decomposition.

Operates on plain numpy arrays so it can be tested standalone against any
returns matrix; data loading and payload assembly live in backend/views/risk.py.
"""

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform


def ledoit_wolf_cov(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Shrunk covariance of a T×N daily-returns matrix.

    Implements Ledoit-Wolf (2004) "Honey, I Shrunk the Sample Covariance
    Matrix": shrinkage of the sample covariance toward the constant-correlation
    target, with the asymptotically optimal intensity estimated from the data.
    Returns (covariance, shrinkage intensity in [0, 1]).
    """
    t, n = returns.shape
    x = returns - returns.mean(axis=0)
    s = x.T @ x / t

    std = np.sqrt(np.diag(s))
    outer_std = np.outer(std, std)
    corr = s / outer_std
    r_bar = (corr.sum() - n) / (n * (n - 1))
    target = r_bar * outer_std
    np.fill_diagonal(target, np.diag(s))

    # pi_hat: sum of asymptotic variances of the sample covariance entries
    y = x**2
    pi_mat = y.T @ y / t - s**2
    pi_hat = pi_mat.sum()

    # rho_hat: asymptotic covariance between sample cov and the CC target.
    # theta_ii[i, j] estimates AsyCov(s_ii, s_ij); theta_jj its mirror.
    m3 = (x**3).T @ x / t
    theta_ii = m3 - np.diag(s)[:, None] * s
    theta_jj = m3.T - np.diag(s)[None, :] * s
    off_diag = ~np.eye(n, dtype=bool)
    rho_hat = np.trace(pi_mat) + r_bar * 0.5 * (
        (np.outer(1.0 / std, std) * theta_ii)[off_diag].sum()
        + (np.outer(std, 1.0 / std) * theta_jj)[off_diag].sum()
    )

    gamma_hat = ((s - target) ** 2).sum()
    if gamma_hat == 0:
        return s, 0.0
    delta = max(0.0, min(1.0, (pi_hat - rho_hat) / gamma_hat / t))
    return delta * target + (1 - delta) * s, delta


def _inverse_variance_weights(cov: np.ndarray) -> np.ndarray:
    ivp = 1.0 / np.diag(cov)
    return ivp / ivp.sum()


def _cluster_variance(cov: np.ndarray, idx: list[int]) -> float:
    sub = cov[np.ix_(idx, idx)]
    w = _inverse_variance_weights(sub)
    return float(w @ sub @ w)


def _correlation_distance(cov: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(std, std), -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)
    return dist


def correlation_clusters(cov: np.ndarray, min_corr: float) -> list[list[int]]:
    """Group assets whose average pairwise correlation is at least min_corr.

    Average-linkage clustering on the correlation distance; returns index
    groups (singletons included), largest first.
    """
    dist = _correlation_distance(cov)
    link = sch.linkage(squareform(dist, checks=False), method="average")
    labels = sch.fcluster(link, t=np.sqrt(0.5 * (1.0 - min_corr)), criterion="distance")
    groups: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        groups.setdefault(label, []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


def hrp_weights(cov: np.ndarray) -> np.ndarray:
    """Hierarchical Risk Parity weights (López de Prado 2016) for a covariance
    matrix. Long-only, sums to 1; needs no expected returns and no matrix
    inversion, so it stays stable for many assets on short histories.
    """
    dist = _correlation_distance(cov)
    link = sch.linkage(squareform(dist, checks=False), method="single")
    order = sch.leaves_list(link).tolist()

    weights = np.ones(cov.shape[0])
    clusters = [order]
    while clusters:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            split = len(cluster) // 2
            left, right = cluster[:split], cluster[split:]
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            next_clusters += [left, right]
        clusters = next_clusters
    return weights / weights.sum()


def risk_contributions(cov: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Fractional risk contributions of each asset (sum to 1) and the portfolio
    daily variance, for weights that sum to 1."""
    port_var = float(weights @ cov @ weights)
    marginal = cov @ weights
    return weights * marginal / port_var, port_var
