"""rate_matches."""

import numpy as np
import pytest

from skillgraph import TS1, BatchResult, Match, Rating


@pytest.mark.slow
def test_equals_manual_loop():
    rng = np.random.default_rng(0)
    env = TS1()
    matches = []
    for _ in range(300):
        k = int(rng.integers(2, 5))
        ids = rng.choice(40, size=2 * k, replace=False).tolist()
        teams = [ids[2 * i : 2 * i + 2] for i in range(k)]
        ranks = [int(x) for x in rng.integers(0, k, size=k)]
        matches.append(Match(teams, ranks, match_weight=float(rng.uniform(0.3, 1))))
    result = env.rate_matches(matches, with_probabilities=True)
    ratings = {}
    for m in matches:
        teams = [[ratings.get(p, env.create_rating()) for p in t] for t in m.teams]
        for ids, new in zip(m.teams, env.rate(teams, m.ranks, None, m.match_weight)):
            ratings.update(zip(ids, new))
    assert result.ratings == ratings
    assert len(result.probabilities) == 300 and all(0 < p <= 1 for p in result.probabilities)


def test_result_type_and_probabilities_off_by_default():
    result = TS1().rate_matches([Match([["a"], ["b"]])])
    assert isinstance(result, BatchResult) and result.probabilities is None
    assert set(result.ratings) == {"a", "b"}


def test_rejects_duplicate_players():
    with pytest.raises(ValueError, match="more than once"):
        TS1().rate_matches([Match([["a", "b"], ["b"]])])


def test_does_not_mutate_input_ratings():
    start = {"a": Rating(30, 2)}
    TS1().rate_matches([Match([["a"], ["b"]])], ratings=start)
    assert start == {"a": Rating(30, 2)}


def test_partial_play_and_big_matches():
    result = TS1().rate_matches(
        [Match([["a", "b"], ["c"]], [0, 1], partial_play=[[1, 0.5], [1]]), Match([[i] for i in range(13)])],
        with_probabilities=True,
    )
    assert result.probabilities[0] is not None and result.probabilities[1] is None  # > 12 teams
