"""quality, win_probabilities, outcome_probability."""

import itertools
import math

import numpy as np
import pytest
from scipy.special import ndtr
from scipy.stats import multivariate_normal

from skillgraph import TS1, Rating


def _rand_teams(rng, k, max_size=3):
    def team():
        size = int(rng.integers(1, max_size + 1))
        return [Rating(float(rng.normal(25, 6)), float(rng.uniform(0.5, 9))) for _ in range(size)]

    return [team() for _ in range(k)]


def test_quality_matches_density_ratio_definition():
    """quality = density of team differences at 0, relative to equal, certain teams."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        env = TS1(beta=float(rng.uniform(1, 8)))
        teams = _rand_teams(rng, int(rng.integers(2, 6)))
        n = len(teams)
        D = np.zeros((n - 1, n))
        for k in range(n - 1):
            D[k, k], D[k, k + 1] = 1, -1
        means = np.array([sum(r.mu for r in t) for t in teams])
        sk = np.diag([sum(r.sigma**2 for r in t) for t in teams])
        bk = np.diag([len(t) * env.beta**2 for t in teams])
        num = multivariate_normal(D @ means, D @ (sk + bk) @ D.T).pdf(np.zeros(n - 1))
        den = multivariate_normal(np.zeros(n - 1), D @ bk @ D.T).pdf(np.zeros(n - 1))
        assert abs(env.quality(teams) - num / den) < 1e-10


def test_quality_ignores_team_order_and_is_bounded():
    rng = np.random.default_rng(1)
    env = TS1()
    for _ in range(100):
        teams = _rand_teams(rng, int(rng.integers(2, 7)))
        q = env.quality(teams)
        assert 0 < q <= 1
        assert abs(env.quality([teams[i] for i in rng.permutation(len(teams))]) - q) < 1e-12
    assert abs(TS1().quality([[Rating(25, 1e-9)], [Rating(25, 1e-9)]]) - 1) < 1e-12


def test_quality_many_teams_no_overflow():
    rng = np.random.default_rng(2)
    teams = [[Rating(float(rng.normal(25, 1)), 1.0)] for _ in range(800)]
    q = TS1(beta=50.0).quality(teams)
    assert math.isfinite(q) and 0 <= q <= 1


def test_win_probabilities_exact_cases():
    env = TS1()
    a, b = [Rating(30, 4)], [Rating(25, 6)]
    p = env.win_probabilities([a, b])
    c = math.sqrt(4**2 + 6**2 + 2 * (env.beta**2 + env.tau**2))
    assert abs(p[0] - ndtr(5 / c)) < 1e-15 and abs(sum(p) - 1) < 1e-15
    same = env.win_probabilities([[Rating()]] * 5)
    assert all(abs(x - 0.2) < 1e-12 for x in same)


@pytest.mark.slow
def test_win_probabilities_match_monte_carlo():
    rng = np.random.default_rng(3)
    env = TS1()
    for _ in range(10):
        teams = _rand_teams(rng, int(rng.integers(3, 8)))
        exact = env.win_probabilities(teams)
        means, varis = env._team_moments(teams, [[1.0] * len(t) for t in teams])
        sims = rng.normal(means, np.sqrt(varis), size=(400_000, len(teams)))
        mc = np.bincount(np.argmax(sims, axis=1), minlength=len(teams)) / 400_000
        assert np.max(np.abs(np.array(exact) - mc)) < 0.004


@pytest.mark.slow
def test_win_probabilities_large_ffa_uses_bounded_monte_carlo():
    teams = [[Rating(float(x), 2.0)] for x in np.random.default_rng(4).normal(25, 5, 400)]
    p = TS1().win_probabilities(teams)
    assert len(p) == 400 and abs(sum(p) - 1) < 1e-12
    assert p == TS1().win_probabilities(teams)  # seeded: reproducible


def test_outcome_probability_two_teams_sums_to_one():
    rng = np.random.default_rng(5)
    for _ in range(200):
        env = TS1(draw_probability=float(rng.uniform(0.001, 0.9)))
        teams = _rand_teams(rng, 2)
        total = sum(env.outcome_probability(teams, r) for r in ([0, 1], [1, 0], [0, 0]))
        assert abs(total - 1) < 1e-12


@pytest.mark.slow
def test_outcome_probability_orderings_sum_to_one_without_draws():
    rng = np.random.default_rng(6)
    env = TS1(draw_probability=0.0)
    for k, tol in ((3, 1e-12), (4, 1e-5)):  # 4+ teams: quasi-Monte Carlo, ~1e-7 per call
        for _ in range(5):
            teams = _rand_teams(rng, k)
            total = sum(env.outcome_probability(teams, list(perm)) for perm in itertools.permutations(range(k)))
            assert abs(total - 1) < tol, (k, total)


def test_outcome_probability_matches_simulation_with_draws():
    rng = np.random.default_rng(7)
    env = TS1(draw_probability=0.3)
    teams = [[Rating(25, 3)], [Rating(27, 5)], [Rating(24, 2)]]
    means, varis = env._team_moments(teams, [[1.0]] * 3)
    sims = rng.normal(means, np.sqrt(varis), size=(2_000_000, 3))
    eps = env.draw_margin(2)
    for ranks in ([0, 0, 1], [1, 0, 2], [0, 1, 1]):
        order = sorted(range(3), key=lambda i: ranks[i])
        ok = np.ones(len(sims), bool)
        for a, b in zip(order, order[1:]):
            d = sims[:, a] - sims[:, b]
            ok &= (np.abs(d) <= eps) if ranks[a] == ranks[b] else (d > eps)
        assert abs(env.outcome_probability(teams, ranks) - ok.mean()) < 4e-3


def test_outcome_log_probability_extreme_upset_is_finite_and_exact():
    env = TS1()
    a, b = [Rating(0, 1)], [Rating(1e4, 1)]
    lp = env.outcome_probability([a, b], [0, 1], log=True)
    c = math.sqrt(2 * (1 + env.beta**2 + env.tau**2))
    x = (-1e4 - env.draw_margin(2)) / c
    import mpmath as mp
    exact = float(mp.log(mp.ncdf(x)))
    assert math.isfinite(lp) and abs(lp - exact) / abs(exact) < 1e-12


def test_outcome_probability_is_deterministic_and_capped():
    env = TS1()
    teams = [[Rating(25, 3)], [Rating(27, 5)], [Rating(24, 2)], [Rating(20, 4)]]
    assert env.outcome_probability(teams, [2, 0, 1, 3]) == env.outcome_probability(teams, [2, 0, 1, 3])
    try:
        env.outcome_probability([[Rating()]] * 13)
    except ValueError:
        return
    raise AssertionError("expected ValueError above 12 teams")
