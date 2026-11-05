"""The engine against independent implementations of the same model."""

import warnings

import numpy as np
import pytest

from cases import max_err, n, rand_case, rand_env
from oracles import OutOfRange, closed_form_two_teams, dense_ep, monte_carlo
from skillgraph import TS1, Rating


def _graph(env, teams, ranks, partial_play):
    """Force the factor-graph path (rate() uses the closed form for 2 teams)."""
    order = sorted(range(len(teams)), key=lambda i: ranks[i])
    ws = partial_play or [[1.0] * len(t) for t in teams]
    g = env._rate_graph([teams[i] for i in order], [ranks[i] for i in order], [ws[i] for i in order], 1.0)
    out = [None] * len(teams)
    for p, i in enumerate(order):
        out[i] = g[p]
    return out


def _sampled_ranks(rng, env, teams):
    """Finishing order drawn from the model itself (realistic upsets only)."""
    perf = [sum(rng.normal(r.mu, np.hypot(r.sigma, env.beta)) for r in t) for t in teams]
    return [int(x) for x in np.argsort(np.argsort(-np.array(perf)))]


def test_two_team_closed_form_equals_factor_graph():
    rng = np.random.default_rng(42)
    for _ in range(n(600, 5000)):
        env = rand_env(rng, min_delta=1e-10, max_iterations=1000)
        teams, ranks, partial_play = rand_case(rng, env, 2)
        assert max_err(env.rate(teams, ranks, partial_play), _graph(env, teams, ranks, partial_play)) < 1e-9


@pytest.mark.slow
def test_two_teams_match_textbook_closed_form():
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(n(600, 5000)):
        env = rand_env(rng)
        teams, ranks, partial_play = rand_case(rng, env, 2)
        try:
            ref = closed_form_two_teams(env, teams, ranks, partial_play)
        except OutOfRange:
            continue
        assert max_err(env.rate(teams, ranks, partial_play), ref) < 1e-8  # oracle's own accuracy
        checked += 1
    assert checked > n(400, 3000)


@pytest.mark.slow
def test_two_teams_match_arbitrary_precision_closed_form():
    rng = np.random.default_rng(4)
    for _ in range(n(150, 1500)):
        env = rand_env(rng)
        teams, ranks, partial_play = rand_case(rng, env, 2)
        ref = closed_form_two_teams(env, teams, ranks, partial_play, force_mp=True)
        assert max_err(env.rate(teams, ranks, partial_play), ref) < 1e-10


def test_two_teams_extremes_match_mpmath_closed_form():
    env = TS1()
    for gap in (40, 400, 4e4):
        for ranks in ([0, 1], [1, 0], [0, 0]):
            teams = [[Rating(25, 2)], [Rating(25 + gap, 2)]]
            ref = closed_form_two_teams(env, teams, ranks)  # falls back to mpmath
            assert max_err(env.rate(teams, ranks), ref) < 1e-8, (gap, ranks)


@pytest.mark.slow
def test_multi_team_matches_dense_ep():
    rng = np.random.default_rng(11)
    checked = 0
    for i in range(n(250, 2000)):
        env = rand_env(rng, min_delta=1e-10, max_iterations=5000)
        teams, ranks, partial_play = rand_case(rng, env, int(rng.integers(3, 9)))
        if i % 2:  # half the cases: realistic finishing orders, which the oracle can always check
            ranks = _sampled_ranks(rng, env, teams)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = env.rate(teams, ranks, partial_play)
        try:
            ref = dense_ep(env, teams, ranks, partial_play)
        except OutOfRange:
            continue
        assert max_err(out, ref) < 1e-8
        checked += 1
    assert checked > n(140, 1100), checked  # guard against a vacuous test


@pytest.mark.slow
def test_large_ffa_matches_dense_ep():
    rng = np.random.default_rng(7)
    configs = [(60, 25 / 3 / 4), (60, 25 / 3 / 20)] + ([(150, 25 / 3 / 4)] if n(0, 1) else [])
    for size, beta in configs:
        env = TS1(beta=beta, draw_probability=0.0, min_delta=1e-10, max_iterations=5000)
        teams = [[Rating(float(rng.normal(25, 6)), float(rng.uniform(0.5, 8.3)))] for _ in range(size)]
        ranks = _sampled_ranks(rng, env, teams)
        assert max_err(env.rate(teams, ranks), dense_ep(env, teams, ranks)) < 1e-8


def test_default_tolerance_is_close_to_converged():
    rng = np.random.default_rng(5)
    for _ in range(n(100, 500)):
        env = rand_env(rng)
        teams, ranks, partial_play = rand_case(rng, env, int(rng.integers(3, 12)))
        tight = TS1(env.mu, env.sigma, env.beta, env.tau, env.draw_probability, min_delta=1e-10, max_iterations=5000)
        assert max_err(env.rate(teams, ranks, partial_play), tight.rate(teams, ranks, partial_play)) < 1e-3


def test_two_teams_equal_exact_posterior_monte_carlo():
    """With two teams, EP is exact: its moments equal the true posterior's."""
    env = TS1()
    teams = [[Rating(30, 4), Rating(22, 6)], [Rating(27, 3), Rating(20, 8)]]
    for ranks in ([0, 1], [1, 0], [0, 0]):
        out = [r for t in env.rate(teams, ranks) for r in t]
        mu, sd, kept = monte_carlo(env, teams, ranks, n=n(1_000_000, 4_000_000))
        for r, m, s in zip(out, mu, sd):
            se_mu, se_sd = s / np.sqrt(kept), s / np.sqrt(2 * kept)  # Monte Carlo standard errors
            assert abs(r.mu - m) < 5 * se_mu and abs(r.sigma - s) < 5 * se_sd, (ranks, r, m, s, kept)


def test_multi_team_close_to_exact_posterior():
    """With 3+ teams EP is an approximation; it should still be close."""
    env = TS1()
    cases = [
        ([[Rating(25, 8)], [Rating(28, 5)], [Rating(20, 7)], [Rating(24, 2)]], [1, 0, 3, 2]),
        ([[Rating(25, 8)], [Rating(28, 5)], [Rating(20, 7)]], [0, 0, 1]),
    ]
    for teams, ranks in cases:
        out = [r for t in env.rate(teams, ranks) for r in t]
        mu, sd, _ = monte_carlo(env, teams, ranks, n=n(1_000_000, 4_000_000))
        for r, m, s in zip(out, mu, sd):
            assert abs(r.mu - m) < 0.15 and abs(r.sigma - s) < 0.15, (r, m, s)
