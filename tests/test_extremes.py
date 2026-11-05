"""Extreme settings: no exceptions, no NaN/inf, valid distributions, sane timing."""

import math
import time
import warnings

import numpy as np
import pytest

from cases import n
from skillgraph import TS1, ConvergenceWarning, Rating

ENVS = {
    "default": TS1(),
    "tiny_beta": TS1(beta=1e-8),
    "huge_beta": TS1(beta=1e8),
    "no_drift": TS1(tau=0),
    "p_draw_1e-12": TS1(draw_probability=1e-12),
    "p_draw_0.999999": TS1(draw_probability=0.999999),
    "tiny_sigma": TS1(sigma=1e-9),
    "huge_sigma": TS1(sigma=1e9),
    "huge_mu": TS1(mu=1e12),
}


def _valid(env, teams, out):
    for t_out, t_in in zip(out, teams):
        for r, r0 in zip(t_out, t_in):
            assert math.isfinite(r.mu) and math.isfinite(r.sigma) and r.sigma > 0, r
            assert r.sigma <= math.hypot(r0.sigma, env.tau) * (1 + 1e-9), (r, r0)


def _exercise(env, teams, ranks, partial_play=None, match_weight=1.0, may_not_converge=False):
    """Must never raise or return invalid ratings. EP must converge unless the
    evidence is absurd (ties across gaps of >= 1e6 sd), where it may warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore" if may_not_converge else "error", ConvergenceWarning)
        out = env.rate(teams, ranks, partial_play, match_weight)
    _valid(env, teams, out)
    q = env.quality(teams, partial_play)
    assert 0.0 <= q <= 1.0 + 1e-12
    wp = env.win_probabilities(teams, partial_play)
    assert all(0.0 <= p <= 1.0 for p in wp) and abs(sum(wp) - 1) < 1e-9
    lp = env.outcome_probability(teams, ranks, partial_play, log=True)
    assert lp <= 1e-12 and not math.isnan(lp)
    return out


@pytest.mark.slow
def test_extreme_grid():
    count = 0
    for env in ENVS.values():
        s = env.sigma
        for gap in (0, 1, 10, 1e3, 1e6, 1e12):
            for k in (1e-6, 1, 1e3):
                a, b = Rating(env.mu, s * k), Rating(env.mu + gap * s, s * k)
                c = Rating(env.mu - gap * s, s)
                absurd = gap >= 1e6
                for ranks in ([0, 1], [1, 0], [0, 0]):
                    _exercise(env, [[a], [b]], ranks)
                    _exercise(env, [[a], [b], [c]], ranks + [2], may_not_converge=absurd)
                    _exercise(env, [[a], [b], [c]], [ranks[0], ranks[1], ranks[0]], may_not_converge=absurd)
                    count += 3
        _exercise(env, [[Rating(env.mu, s)]] * 3, [0, 0, 0])
        _exercise(env, [[Rating(env.mu, s)] * 2, [Rating(env.mu, s)]], [0, 1], [[1e-9, 1.0], [1e-9]])
        for mw in (0.0, 1e-9, 0.5):
            three = [[Rating(env.mu, s)], [Rating(env.mu + 5 * s, s)], [Rating(env.mu, s)]]
            _exercise(env, three, [0, 1, 2], match_weight=mw)
            _exercise(env, [[Rating(env.mu, s)], [Rating(env.mu + 5 * s, s)]], [0, 1], match_weight=mw)
        count += 8
    assert count > 1000


def test_every_tie_pattern_converges_up_to_large_gaps():
    import itertools

    for env in ENVS.values():
        s = env.sigma
        for gap in (3, 100, 1e4, 1e5):
            for k in (1e-6, 1, 1e3):
                teams = [[Rating(env.mu, s * k)], [Rating(env.mu + gap * s, s * k)], [Rating(env.mu - gap * s, s)]]
                for ranks in itertools.product(range(3), repeat=3):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        _valid(env, teams, env.rate(teams, list(ranks)))


def test_huge_matches_are_fast_and_valid():
    env = TS1()
    rng = np.random.default_rng(0)
    cases = [
        ([[Rating()] * 500, [Rating()] * 500], 5.0),
        ([[Rating(float(x), 3.0)] for x in rng.normal(25, 8, n(1000, 3000))], n(20.0, 60.0)),
        ([[Rating()] * 5 for _ in range(30)], 10.0),
    ]
    for teams, budget in cases:
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = env.rate(teams)
        assert time.time() - t0 < budget, (len(teams), time.time() - t0)
        _valid(env, teams, out)
    assert math.isfinite(env.quality([[Rating(float(x), 2.0)] for x in rng.normal(25, 8, 600)]))


def test_long_sequences_stay_finite():
    """Many updates on one pair: sigma shrinks toward the drift floor, never to 0."""
    for env in (TS1(), TS1(tau=0), TS1(beta=1e-6)):
        a, b = Rating(), Rating()
        for i in range(n(3000, 30000)):
            a, b = env.rate_1vs1(a, b, drawn=(i % 3 == 0)) if i % 2 else env.rate_1vs1(b, a)[::-1]
        for r in (a, b):
            assert math.isfinite(r.mu) and 0 < r.sigma < env.sigma
