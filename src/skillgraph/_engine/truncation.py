"""Moment matching for truncated Gaussians (the only non-Gaussian step in TrueSkill).

For a standardized performance difference ``t`` and standardized draw margin
``eps`` these return ``(v, omw)``:

- ``v``: shift of the standardized mean after conditioning on the outcome;
- ``omw`` = ``1 - w``: ratio of new to old variance.

The textbook formulas compute ``w`` and then ``1 - w``. That cancels
catastrophically in the tails (big upsets) and for narrow draw margins (small
draw probabilities), giving variance errors of 100% or more. Here each regime
uses a formulation accurate to ~1e-10 relative or better, verified against
arbitrary-precision (mpmath) references in the test suite:

- win, x = t - eps >= -10: log-space Mills ratio;
- win, x < -10: asymptotic series whose cancelling terms were removed exactly
  (integer coefficients), so no floating-point cancellation remains;
- draw, small tilt: Gauss-Legendre quadrature in interval coordinates
  (handles arbitrarily narrow margins);
- draw, interval far in a tail: reduces to the one-sided (win) case;
- draw, interval containing the mode: direct formula (well conditioned there).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.polynomial import polyval
from scipy.special import log_ndtr, ndtr

from .._constants import DRAW_QUAD_LIMIT, DRAW_QUAD_NODES, SERIES_BELOW_X, SERIES_TERMS

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_TINY = float(np.finfo(float).tiny)


def _pdf(x: float) -> float:
    return math.exp(-0.5 * x * x - _LOG_SQRT_2PI)


# ---------------------------------------------------------------------------
# Asymptotic series for x -> -inf (y = -x, u = 1/y^2).
# Mills ratio R(y) = (1/y) * S(u),  S(u) = sum_n (-1)^n (2n-1)!! u^n.
# v = 1/R = y / S.  1 - w = Num(u) / (u * S^2) with Num = u*S^2 - (1 - S),
# whose u^1 coefficient cancels exactly; we drop it in integer arithmetic.
def _series_coeffs():
    s = [1]
    for n in range(1, SERIES_TERMS + 1):
        s.append(-s[-1] * (2 * n - 1))
    s_sq = [sum(s[i] * s[k - i] for i in range(k + 1)) for k in range(SERIES_TERMS + 1)]
    # Num_k = [u^(k-1)] S^2 + s_k for k >= 1; Num_1 == 0 exactly.
    num = [s_sq[k - 1] + s[k] for k in range(1, SERIES_TERMS + 1)]
    assert num[0] == 0
    num_over_u2 = num[1:]  # coefficients of u^0, u^1, ... of Num(u)/u^2
    return np.array(s, dtype=float), np.array(num_over_u2, dtype=float)


_S_COEFFS, _NUM_COEFFS = _series_coeffs()


def _win_moments_x(x: float) -> tuple[float, float]:
    """(v, 1-w) for N(0,1) conditioned on z > -x, i.e. v = pdf(x)/cdf(x)."""
    if x < SERIES_BELOW_X:
        y = -x
        u = 1.0 / (y * y)
        s = float(polyval(u, _S_COEFFS))
        return y / s, u * float(polyval(u, _NUM_COEFFS)) / (s * s)
    v = math.exp(-0.5 * x * x - _LOG_SQRT_2PI - float(log_ndtr(x)))
    omw = 1.0 - v * (v + x)
    return v, min(max(omw, _TINY), 1.0)


def win(t: float, eps: float) -> tuple[float, float]:
    """Condition on a win: difference > margin."""
    return _win_moments_x(t - eps)


# ---------------------------------------------------------------------------
_GL_X, _GL_W = np.polynomial.legendre.leggauss(DRAW_QUAD_NODES)


def _quad_moments(kappa: float, lam: float) -> tuple[float, float]:
    """Mean and variance of u on [-1, 1] with density ∝ exp(kappa*u - lam*u^2)."""
    e = kappa * _GL_X - lam * _GL_X * _GL_X
    f = _GL_W * np.exp(e - e.max())
    z = f.sum()
    mean = 0.0 if kappa == 0.0 else float((f * _GL_X).sum() / z)  # exact 0 by symmetry
    var = float((f * (_GL_X - mean) ** 2).sum() / z)
    return mean, var


def draw(t: float, eps: float) -> tuple[float, float]:
    """Condition on a draw: |difference| <= margin."""
    if not eps > 0.0:
        raise ValueError("a draw needs a positive margin (draw_probability > 0)")
    # Solve for |t| and mirror, so draw(-t) is exactly -draw(t).
    sign = -1.0 if t < 0 else 1.0
    a = abs(t)
    kappa = a * eps
    lam = 0.5 * eps * eps
    if kappa <= DRAW_QUAD_LIMIT and lam <= DRAW_QUAD_LIMIT:
        # d ~ N(a, 1) truncated to [-eps, eps]; write d = eps * u.
        mean_u, var_u = _quad_moments(kappa, lam)
        return sign * (eps * mean_u - a), max(eps * eps * var_u, _TINY)
    if a >= eps:
        # Interval lies in a tail; the far end carries relative mass
        # < exp(-2*kappa) < exp(-100), so this is a one-sided truncation.
        v, omw = _win_moments_x(eps - a)
        return -sign * v, omw
    # Interval contains the mode and is wide: direct formula is well conditioned.
    lo, hi = -eps - a, eps - a
    z = float(ndtr(hi) - ndtr(lo))
    p_lo, p_hi = _pdf(lo), _pdf(hi)
    mean = (p_lo - p_hi) / z
    second = 1.0 + (lo * p_lo - hi * p_hi) / z
    return sign * mean, max(second - mean * mean, _TINY)
