"""Probabilities of performance orderings for independent Gaussian teams."""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad
from scipy.special import log_ndtr, ndtr
from scipy.stats import multivariate_normal

from .._constants import (
    MC_CHUNK_ELEMENTS,
    MVN_ABSEPS,
    MVN_MAXPTS_PER_DIM,
    MVN_RELEPS,
    NEGLIGIBLE_Z,
    WIN_PROB_SPAN,
    WIN_PROB_STEP_OFFSETS,
)

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_STEP_OFFSETS = np.array([WIN_PROB_STEP_OFFSETS])


def log_box(lo: float, hi: float) -> float:
    """log(Phi(hi) - Phi(lo)), stable in both tails."""
    if hi <= 0:
        a, b = float(log_ndtr(hi)), float(log_ndtr(lo))
    elif lo >= 0:
        a, b = float(log_ndtr(-lo)), float(log_ndtr(-hi))
    else:
        return math.log(float(ndtr(hi) - ndtr(lo)))
    return a + math.log1p(-math.exp(b - a)) if b < a else -math.inf


def best_probability(i: int, m: np.ndarray, sd: np.ndarray) -> float:
    """P(T_i > T_j for all j != i), with T_j ~ N(m_j, sd_j^2) independent.

    Integrates over z = (T_i - m_i)/sd_i. Each other team contributes a step
    Phi(a_j + b_j z) of width 1/b_j; breakpoints around every step stop the
    adaptive quadrature from stepping over narrow transitions.
    """
    others = np.arange(len(m)) != i
    a = (m[i] - m[others]) / sd[others]
    b = sd[i] / sd[others]
    if np.any(a + b * WIN_PROB_SPAN < NEGLIGIBLE_Z):
        return 0.0

    def integrand(z: float) -> float:
        return math.exp(-0.5 * z * z - _LOG_SQRT_2PI + float(np.sum(log_ndtr(a + b * z))))

    pts = ((-a / b)[:, None] + (1.0 / b)[:, None] * _STEP_OFFSETS).ravel()
    pts = np.unique(pts[(pts > -WIN_PROB_SPAN) & (pts < WIN_PROB_SPAN)])
    val, _ = quad(
        integrand,
        -WIN_PROB_SPAN,
        WIN_PROB_SPAN,
        points=pts if len(pts) else None,
        limit=max(200, 2 * len(pts) + 100),
        epsabs=1e-15,
        epsrel=1e-11,
    )
    return max(float(val), 0.0)


def monte_carlo_best(m: np.ndarray, sd: np.ndarray, n_samples: int, seed: int | None) -> list[float]:
    """Seeded Monte Carlo estimate of P(each team is best), in bounded memory."""
    n = len(m)
    rng = np.random.default_rng(seed)
    counts = np.zeros(n, dtype=np.int64)
    chunk = max(1, min(n_samples, MC_CHUNK_ELEMENTS // n))
    done = 0
    while done < n_samples:
        k = min(chunk, n_samples - done)
        sims = rng.standard_normal((k, n)) * sd + m
        counts += np.bincount(np.argmax(sims, axis=1), minlength=n)
        done += k
    return (counts / n_samples).tolist()


def chain_box_probability(
    means: list[float], variances: list[float], lower: np.ndarray, upper: np.ndarray
) -> float:
    """P(lower_k <= T_k - T_{k+1} <= upper_k for all k), T_k independent Gaussians.

    Deterministic (seeded) quasi-Monte Carlo; absolute error ~1e-7.
    """
    k = len(means) - 1
    mean = np.array([means[i] - means[i + 1] for i in range(k)])
    cov = np.zeros((k, k))
    for i in range(k):
        cov[i, i] = variances[i] + variances[i + 1]
        if i + 1 < k:
            cov[i, i + 1] = cov[i + 1, i] = -variances[i + 1]
    dist = multivariate_normal(
        mean=mean, cov=cov, seed=0, abseps=MVN_ABSEPS, releps=MVN_RELEPS, maxpts=MVN_MAXPTS_PER_DIM * k
    )
    return min(max(float(dist.cdf(upper, lower_limit=lower)), 0.0), 1.0)
