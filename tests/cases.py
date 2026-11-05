"""Random environments/matches and comparison helpers shared by the tests."""

import os

from skillgraph import TS1, Rating

STRESS = os.environ.get("SKILLGRAPH_STRESS") == "1"  # heavier runs when set


def n(default, stress):
    return stress if STRESS else default


def rand_env(rng, **overrides):
    sigma = float(10 ** rng.uniform(-0.5, 1.5))
    kw = dict(
        mu=float(rng.normal(0, 50)),
        sigma=sigma,
        beta=float(sigma * 10 ** rng.uniform(-1.5, 0.5)),
        tau=float(sigma * 10 ** rng.uniform(-4, -0.5)),
        draw_probability=float(rng.uniform(0.001, 0.6)),
    )
    kw.update(overrides)
    return TS1(**kw)


def rand_case(rng, env, n_teams, max_size=4, weighted=True, distinct_ranks=False):
    teams = [
        [
            Rating(float(env.mu + rng.normal(0, 3 * env.sigma)), float(env.sigma * 10 ** rng.uniform(-1.5, 0.3)))
            for _ in range(int(rng.integers(1, max_size + 1)))
        ]
        for _ in range(n_teams)
    ]
    if distinct_ranks:
        ranks = [int(x) for x in rng.permutation(n_teams)]
    else:
        ranks = [int(x) for x in rng.integers(0, n_teams, size=n_teams)]
    partial_play = None
    if weighted and rng.random() < 0.5:
        partial_play = [[float(rng.uniform(0.05, 1)) for _ in t] for t in teams]
    return teams, ranks, partial_play


def max_err(out, ref):
    """Largest |mu| or |sigma| difference, in units of the output sigma.

    ``ref`` may hold Ratings or (mu, sigma) tuples.
    """
    worst = 0.0
    for ta, tb in zip(out, ref):
        for a, b in zip(ta, tb):
            bm, bs = (b.mu, b.sigma) if isinstance(b, Rating) else b
            worst = max(worst, abs(a.mu - bm) / a.sigma, abs(a.sigma - bs) / a.sigma)
    return worst
