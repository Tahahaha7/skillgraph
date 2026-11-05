"""Bayesian skill ratings via expectation propagation.

    import skillgraph as sg
    sg.rate(teams, ranks)        # TrueSkill 1, default settings (same as sg.TS1.rate)
    sg.TS1(beta=4.0).rate(...)   # TrueSkill 1, custom settings
"""

from ._types import BatchResult, ConvergenceWarning, Match, Rating
from .ts1 import TS1

__version__ = "0.1.0.dev0"

# Shortcuts: TrueSkill 1 with default settings.
rate = TS1.rate
rate_1vs1 = TS1.rate_1vs1
rate_matches = TS1.rate_matches
quality = TS1.quality
win_probabilities = TS1.win_probabilities
outcome_probability = TS1.outcome_probability

__all__ = [
    "TS1",
    "BatchResult",
    "ConvergenceWarning",
    "Match",
    "Rating",
    "outcome_probability",
    "quality",
    "rate",
    "rate_1vs1",
    "rate_matches",
    "win_probabilities",
]
