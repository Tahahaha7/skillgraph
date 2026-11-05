"""Invariants that must hold for every input (randomized)."""

import math
import warnings

import numpy as np

from cases import max_err, n, rand_case, rand_env
from skillgraph import TS1, Rating


def test_team_order_does_not_matter():
    rng = np.random.default_rng(1)
    for _ in range(n(300, 3000)):
        env = rand_env(rng)
        teams, ranks, partial_play = rand_case(rng, env, int(rng.integers(2, 7)), distinct_ranks=True)
        perm = rng.permutation(len(teams))
        out = env.rate(teams, ranks, partial_play)
        pp_p = [partial_play[i] for i in perm] if partial_play else None
        out_p = env.rate([teams[i] for i in perm], [ranks[i] for i in perm], pp_p)
        assert max_err([out[i] for i in perm], out_p) < 1e-12


def test_scale_invariance():
    rng = np.random.default_rng(2)
    for _ in range(n(200, 2000)):
        env = rand_env(rng)
        teams, ranks, partial_play = rand_case(rng, env, int(rng.integers(2, 6)))
        out = env.rate(teams, ranks, partial_play)
        for k in (1e-3, 7.3, 1e4):
            env_k = TS1(env.mu * k, env.sigma * k, env.beta * k, env.tau * k, env.draw_probability)
            teams_k = [[Rating(r.mu * k, r.sigma * k) for r in t] for t in teams]
            out_k = env_k.rate(teams_k, ranks, partial_play)
            scaled = [[Rating(r.mu * k, r.sigma * k) for r in t] for t in out]
            assert max_err(out_k, scaled) < 1e-8


def test_translation_invariance_for_equal_size_teams():
    rng = np.random.default_rng(3)
    for _ in range(n(200, 2000)):
        env = rand_env(rng)
        size, k = int(rng.integers(1, 4)), int(rng.integers(2, 6))
        def player():
            return Rating(float(rng.normal(0, 3 * env.sigma)), float(env.sigma * rng.uniform(0.1, 1.5)))

        teams = [[player() for _ in range(size)] for _ in range(k)]
        ranks = [int(x) for x in rng.integers(0, k, size=k)]
        out = env.rate(teams, ranks)
        for c in (-1e4 * env.sigma, 3.7 * env.sigma, 1e5 * env.sigma):
            shifted = env.rate([[Rating(r.mu + c, r.sigma) for r in t] for t in teams], ranks)
            back = [[Rating(r.mu - c, r.sigma) for r in t] for t in shifted]
            assert max_err(out, back) < 1e-6


def test_winners_go_up_losers_go_down():
    rng = np.random.default_rng(4)
    for _ in range(n(500, 5000)):
        env = rand_env(rng)
        k = int(rng.integers(2, 7))
        teams, ranks, partial_play = rand_case(rng, env, k, distinct_ranks=True)
        out = env.rate(teams, ranks, partial_play)
        first, last = ranks.index(0), ranks.index(k - 1)
        # (allow 1e-9 sigma: with a foregone result the true update is ~0)
        assert all(a.mu >= b.mu - 1e-9 * b.sigma for a, b in zip(out[first], teams[first]))
        assert all(a.mu <= b.mu + 1e-9 * b.sigma for a, b in zip(out[last], teams[last]))


def test_draws_pull_teams_together():
    rng = np.random.default_rng(5)
    for _ in range(n(500, 5000)):
        env = rand_env(rng)
        teams, _, partial_play = rand_case(rng, env, 2)
        out = env.rate(teams, [0, 0], partial_play)
        w = partial_play or [[1.0] * len(t) for t in teams]
        strength = [sum(x * r.mu for x, r in zip(wt, t)) for wt, t in zip(w, teams)]
        fav, dog = (0, 1) if strength[0] > strength[1] else (1, 0)
        assert all(a.mu <= b.mu + 1e-9 * b.sigma for a, b in zip(out[fav], teams[fav]))
        assert all(a.mu >= b.mu - 1e-9 * b.sigma for a, b in zip(out[dog], teams[dog]))


def test_uncertainty_never_grows_beyond_drift():
    rng = np.random.default_rng(6)
    for _ in range(n(500, 5000)):
        env = rand_env(rng)
        teams, ranks, partial_play = rand_case(rng, env, int(rng.integers(2, 7)))
        for t_out, t_in in zip(env.rate(teams, ranks, partial_play), teams):
            for a, b in zip(t_out, t_in):
                assert 0 < a.sigma <= math.sqrt(b.sigma**2 + env.tau**2) * (1 + 1e-12)


def test_draw_between_identical_players_keeps_means_exactly():
    for env in (TS1(), TS1(beta=1e-3), TS1(draw_probability=1e-9), TS1(draw_probability=0.9)):
        for r in (Rating(), Rating(-40, 0.01), Rating(1e6, 300)):
            a, b = env.rate_1vs1(r, r, drawn=True)
            assert a.mu == r.mu and b.mu == r.mu and a.sigma == b.sigma


def test_mirror_matches_are_exact_mirrors():
    """Negating every rating and the result gives exactly negated means."""
    env = TS1()
    a, b = Rating(30, 4), Rating(20, 6)
    new_a, new_b = env.rate_1vs1(a, b)                                  # a beats b
    (neg_a,), (neg_b,) = env.rate([[Rating(-30, 4)], [Rating(-20, 6)]], [1, 0])  # -a loses to -b
    assert new_a.mu == -neg_a.mu and new_b.mu == -neg_b.mu
    assert new_a.sigma == neg_a.sigma and new_b.sigma == neg_b.sigma


def test_bigger_upsets_move_ratings_more():
    env = TS1()
    gains = [env.rate_1vs1(Rating(25 - gap, 5), Rating(25, 5))[0].mu - (25 - gap) for gap in np.linspace(-30, 60, 50)]
    assert all(x < y for x, y in zip(gains, gains[1:]))


def test_deterministic_and_inputs_untouched():
    rng = np.random.default_rng(7)
    env = rand_env(rng)
    teams, ranks, partial_play = rand_case(rng, env, 5)
    snapshot = repr((teams, ranks, partial_play))
    a, b = env.rate(teams, ranks, partial_play), env.rate(teams, ranks, partial_play)
    assert repr(a) == repr(b) and max_err(a, b) == 0
    assert repr((teams, ranks, partial_play)) == snapshot


def test_full_partial_play_equals_none():
    rng = np.random.default_rng(8)
    for _ in range(n(100, 1000)):
        env = rand_env(rng)
        teams, ranks, _ = rand_case(rng, env, int(rng.integers(2, 6)), weighted=False)
        ones = [[1.0] * len(t) for t in teams]
        assert max_err(env.rate(teams, ranks), env.rate(teams, ranks, ones)) == 0


def test_partial_play_learns_less():
    env = TS1()
    (full, half), _ = env.rate([[Rating(), Rating()], [Rating(), Rating()]], partial_play=[[1.0, 0.5], [1.0, 1.0]])
    assert 25 < half.mu < full.mu and full.sigma < half.sigma < Rating().sigma
    tiny = env.rate([[Rating(), Rating()], [Rating(), Rating()]], partial_play=[[1.0, 1e-9], [1.0, 1.0]])[0][1]
    assert abs(tiny.mu - 25) < 1e-6 and abs(tiny.sigma - math.hypot(Rating().sigma, env.tau)) < 1e-6


def test_match_weight():
    rng = np.random.default_rng(9)
    for _ in range(n(200, 2000)):
        env = rand_env(rng)
        teams, ranks, partial_play = rand_case(rng, env, int(rng.integers(2, 6)))
        full = env.rate(teams, ranks, partial_play)
        assert max_err(full, env.rate(teams, ranks, partial_play, match_weight=1.0)) == 0
        none = env.rate(teams, ranks, partial_play, match_weight=0.0)
        drift = [[(r.mu, math.hypot(r.sigma, env.tau)) for r in t] for t in teams]
        assert max_err(none, drift) < 1e-12
        prev = none
        for alpha in (0.25, 0.5, 0.75, 1.0):
            cur = env.rate(teams, ranks, partial_play, match_weight=alpha)
            for tc, tp, t0 in zip(cur, prev, teams):
                for c, p, r0 in zip(tc, tp, t0):
                    assert abs(c.mu - r0.mu) >= abs(p.mu - r0.mu) - 1e-9 * r0.sigma
                    assert c.sigma <= p.sigma * (1 + 1e-12)
            prev = cur


def test_no_convergence_warnings_on_random_matches():
    rng = np.random.default_rng(10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(n(300, 3000)):
            env = rand_env(rng)
            teams, ranks, partial_play = rand_case(rng, env, int(rng.integers(3, 20)))
            env.rate(teams, ranks, partial_play)
