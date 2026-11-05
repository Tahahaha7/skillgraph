"""Invalid inputs raise clear errors; valid unusual inputs are accepted."""

import math

import numpy as np
import pytest

from cases import max_err
from skillgraph import TS1, Rating

NAN, INF = math.nan, math.inf
TWO = [[Rating()], [Rating()]]


def test_invalid_ratings():
    for mu, sigma in [(NAN, 1), (INF, 1), (25, 0), (25, -1), (25, NAN), (25, INF)]:
        with pytest.raises(ValueError):
            Rating(mu, sigma)


def test_invalid_model_settings():
    for kwargs in [
        {"sigma": 0}, {"beta": -1}, {"tau": -1}, {"tau": NAN}, {"mu": INF},
        {"draw_probability": 1.0}, {"draw_probability": -0.1}, {"max_iterations": 0}, {"min_delta": 0},
    ]:
        with pytest.raises(ValueError):
            TS1(**kwargs)


def test_invalid_rate_arguments():
    env = TS1()
    for kwargs in [
        {"teams": [[Rating()]]},
        {"teams": [[Rating()], []]},
        {"teams": TWO, "ranks": [0]},
        {"teams": TWO, "ranks": [0, NAN]},
        {"teams": TWO, "partial_play": [[0.0], [1.0]]},
        {"teams": TWO, "partial_play": [[NAN], [1.0]]},
        {"teams": TWO, "partial_play": [[1.5], [1.0]]},
        {"teams": TWO, "partial_play": [[1.0]]},
        {"teams": TWO, "match_weight": 1.5},
        {"teams": TWO, "match_weight": NAN},
    ]:
        with pytest.raises(ValueError):
            env.rate(**kwargs)
    with pytest.raises(TypeError):
        env.rate([[Rating()], [(25, 8)]])
    with pytest.raises(ValueError, match="draw_probability is 0"):
        TS1(draw_probability=0.0).rate_1vs1(Rating(), Rating(), drawn=True)


def test_accepts_numpy_and_tuples():
    env = TS1()
    out = env.rate(((Rating(np.float64(25), np.float32(8)),), (Rating(np.int64(30), 5),)), ranks=np.array([1, 0]))
    ref = env.rate([[Rating(25, float(np.float32(8)))], [Rating(30, 5)]], [1, 0])
    assert max_err(out, ref) == 0 and isinstance(out[0][0].mu, float)
