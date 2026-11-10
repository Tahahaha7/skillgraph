"""Regenerate tests/golden_ts1.json (inputs + expected outputs).

Only run this when a change in results is intended; note it in CHANGELOG.md.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from cases import rand_case, rand_env  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(2026)
    cases = []
    for i in range(300):
        env = rand_env(rng)
        teams, ranks, pp = rand_case(rng, env, 2 if i < 150 else int(rng.integers(3, 9)))
        mw = 1.0 if i % 5 else float(rng.uniform(0, 1))
        cases.append({
            "model": {"mu": env.mu, "sigma": env.sigma, "beta": env.beta, "tau": env.tau,
                      "draw_probability": env.draw_probability},
            "teams": [[[r.mu, r.sigma] for r in t] for t in teams],
            "ranks": ranks,
            "partial_play": pp,
            "match_weight": mw,
            "expected": {
                "ratings": [[[r.mu, r.sigma] for r in t] for t in env.rate(teams, ranks, pp, mw)],
                "quality": env.quality(teams, pp),
                "win_probabilities": env.win_probabilities(teams, pp),
                "outcome_log_probability": env.outcome_probability(teams, ranks, pp, log=True),
            },
        })
    path = ROOT / "tests" / "golden_ts1.json"
    path.write_text(json.dumps({"format": 1, "cases": cases}))
    print(f"wrote {len(cases)} cases to {path}")


if __name__ == "__main__":
    main()
