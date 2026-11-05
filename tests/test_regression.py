"""Results must not drift between versions (cf. upstream issue #33).

golden_ts1.json holds inputs and expected outputs. If a change is
intentional, regenerate it with ``python scripts/make_golden.py`` and say so
in the changelog.
"""

import json
import math
from pathlib import Path

from skillgraph import TS1, Rating

GOLDEN = json.loads((Path(__file__).parent / "golden_ts1.json").read_text())["cases"]


def _inputs(case):
    teams = [[Rating(mu, sigma) for mu, sigma in t] for t in case["teams"]]
    return TS1(**case["model"]), teams, case["ranks"], case["partial_play"]


def test_ratings():
    for case in GOLDEN:
        env, teams, ranks, pp = _inputs(case)
        for t_out, t_exp in zip(env.rate(teams, ranks, pp, case["match_weight"]), case["expected"]["ratings"]):
            for r, (mu, sigma) in zip(t_out, t_exp):
                assert abs(r.mu - mu) <= 1e-9 * sigma and abs(r.sigma - sigma) <= 1e-9 * sigma


def test_predictions():
    for case in GOLDEN:
        env, teams, ranks, pp = _inputs(case)
        exp = case["expected"]
        assert abs(env.quality(teams, pp) - exp["quality"]) <= 1e-12
        assert all(abs(a - b) <= 1e-12 for a, b in zip(env.win_probabilities(teams, pp), exp["win_probabilities"]))
        lp, lp_exp = env.outcome_probability(teams, ranks, pp, log=True), exp["outcome_log_probability"]
        if lp == lp_exp:  # includes both -inf
            continue
        if len(teams) == 2:
            assert abs(lp - lp_exp) <= 1e-10 * max(1.0, abs(lp_exp))
        else:  # 3+ teams: compare probabilities at the method's ~1e-7 accuracy
            assert abs(math.exp(lp) - math.exp(lp_exp)) <= 1e-6
