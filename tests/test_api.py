"""Public API: sg.rate, sg.TS1.rate and sg.TS1(...).rate behave consistently."""

import inspect

import numpy as np

import skillgraph as sg
from cases import max_err
from skillgraph import TS1, Match, Rating

TEAMS = [[Rating(27, 5), Rating(22, 7)], [Rating(25, 6)], [Rating(30, 3)]]
RANKS = [1, 0, 2]


def test_call_styles_agree():
    ref = TS1().rate(TEAMS, RANKS)
    for out in (sg.rate(TEAMS, RANKS), TS1.rate(TEAMS, RANKS), sg.rate(teams=TEAMS, ranks=RANKS)):
        assert max_err(out, ref) == 0


def test_custom_instance_differs_from_default():
    assert max_err(TS1(beta=1.0).rate(TEAMS, RANKS), sg.rate(TEAMS, RANKS)) > 1e-3


def test_every_shortcut_matches_class_and_instance():
    two = [[Rating(27, 5)], [Rating(25, 6)]]
    calls = {
        "rate_1vs1": (Rating(), Rating(30, 4)),
        "rate_matches": ([Match([["a"], ["b"]], [0, 1]), Match([["b"], ["c"]], [0, 0])],),
        "quality": (TEAMS,),
        "win_probabilities": (two,),
        "outcome_probability": (TEAMS, RANKS),
    }
    for name, args in calls.items():
        assert repr(getattr(sg, name)(*args)) == repr(getattr(TS1, name)(*args)) == repr(getattr(TS1(), name)(*args))
    assert repr(TS1.create_rating()) == repr(TS1().create_rating()) == repr(Rating())


def test_public_names():
    assert set(sg.__all__) == {
        "TS1", "BatchResult", "ConvergenceWarning", "Match", "Rating",
        "outcome_probability", "quality", "rate", "rate_1vs1", "rate_matches", "win_probabilities",
    }
    assert all(hasattr(sg, name) for name in sg.__all__)


def test_signatures_and_docs():
    for fn in (sg.rate, TS1.rate, TS1().rate):
        assert next(iter(inspect.signature(fn).parameters)) == "teams"
        assert "Update ratings" in fn.__doc__


def test_shared_default_is_never_mutated():
    d = TS1._default()
    TS1.rate(TEAMS, RANKS)
    sg.rate(TEAMS, RANKS)
    assert TS1._default() is d and repr(d) == repr(TS1())


def test_subclass_gets_its_own_default():
    class Tuned(TS1):
        def __init__(self):
            super().__init__(beta=2.0)

    assert Tuned._default().beta == 2.0 and TS1._default().beta == TS1().beta
    assert max_err(Tuned.rate(TEAMS, RANKS), TS1(beta=2.0).rate(TEAMS, RANKS)) == 0


def test_numpy_ranks():
    assert max_err(sg.rate(TEAMS, np.array(RANKS)), TS1().rate(TEAMS, RANKS)) == 0
