"""Truncated-Gaussian moment matching vs arbitrary-precision mpmath."""

import math

import mpmath as mp
import numpy as np
import pytest

from cases import n
from oracles import mp_draw, mp_win
from skillgraph._engine.truncation import _NUM_COEFFS, _S_COEFFS, draw, win


def _rel(a, b):
    b = mp.mpf(b)
    return float(abs(mp.mpf(a) - b) / abs(b))


def test_win_matches_mpmath_everywhere():
    xs = list(np.linspace(-60, 40, n(401, 2001))) + [-1e3, -1e5, -1e8, -1e12, -10.0, -10.000001, -9.999999]
    for x in xs:
        V, omw = mp_win(x)
        v, o = win(float(x), 0.0)
        if V > 1e-300:
            assert _rel(v, V) < 1e-12, x
        assert _rel(o, omw) < 2e-10, x  # the variance ratio: the hard part
        assert 0 < o <= 1


@pytest.mark.slow
def test_draw_matches_mpmath_everywhere():
    rng = np.random.default_rng(1)
    grid = [(t, e) for t in [0, 1e-3, 0.3, 1, 3, 7, 10, 20, 40, 100, 1e3, 1e5]
            for e in [1e-12, 1e-9, 1e-6, 1e-3, 0.1, 0.5, 1, 3, 7, 10, 20, 50]]
    grid += [(float(rng.uniform(-60, 60)), float(10 ** rng.uniform(-10, 1.7))) for _ in range(n(60, 400))]
    for t, e in grid:
        for s in (1, -1):
            m, var = mp_draw(s * t, e)
            v, o = draw(s * t, e)
            assert _rel(o, var) < 2e-10, (s * t, e)
            # mean error measured in posterior standard deviations
            assert float(abs(mp.mpf(v) - m) / mp.sqrt(var)) < 1e-3, (s * t, e)
            if abs(m) > 1e-6:
                assert _rel(v, m) < 1e-10, (s * t, e)


def test_draw_symmetry():
    for t in (0.0, 0.7, 5.0, 80.0):
        for e in (1e-8, 0.3, 4.0, 30.0):
            v1, o1 = draw(t, e)
            v2, o2 = draw(-t, e)
            assert v1 == -v2 and o1 == o2


def test_textbook_formulas_would_fail_here():
    """Regression guard: the textbook path (float Mills ratio, then 1 - v(v+x)),
    as used by common implementations, is badly wrong here; ours is not."""
    from scipy.special import log_ndtr

    for x in (-1e3, -1e5):
        _, omw = mp_win(x)
        v = math.exp(-0.5 * x * x - 0.5 * math.log(2 * math.pi) - float(log_ndtr(x)))
        assert _rel(1 - v * (v + x), omw) > 0.5
        assert _rel(win(x, 0.0)[1], omw) < 1e-10


def test_series_coefficients_exact():
    # S(u) = sum (-1)^n (2n-1)!! u^n ; Num/u^2 starts 1, -8, ...
    assert list(_S_COEFFS[:5]) == [1, -1, 3, -15, 105]
    assert list(_NUM_COEFFS[:2]) == [1, -8]


def test_draw_requires_positive_margin():
    with pytest.raises(ValueError):
        draw(3.0, 0.0)
