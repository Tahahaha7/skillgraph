"""Data types shared by all models."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

DEFAULT_MU = 25.0
DEFAULT_SIGMA = DEFAULT_MU / 3


class ConvergenceWarning(UserWarning):
    """Message passing did not converge, even with damping."""


@dataclass(frozen=True)
class Rating:
    """A player's skill belief, N(mu, sigma^2)."""

    mu: float = DEFAULT_MU
    sigma: float = DEFAULT_SIGMA

    def __post_init__(self) -> None:
        mu, sigma = float(self.mu), float(self.sigma)
        if not math.isfinite(mu):
            raise ValueError(f"mu must be finite, got {self.mu!r}")
        if not (math.isfinite(sigma) and sigma > 0):
            raise ValueError(f"sigma must be positive and finite, got {self.sigma!r}")
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "sigma", sigma)

    def conservative(self, k: float = 3.0) -> float:
        """Leaderboard score mu - k*sigma (TrueSkill's usual k=3)."""
        return self.mu - k * self.sigma

    def __repr__(self) -> str:
        return f"Rating(mu={self.mu:.3f}, sigma={self.sigma:.3f})"


Team = Sequence[Rating]


@dataclass(frozen=True)
class Match:
    """One match between teams of player ids, for ``rate_matches``."""

    teams: Sequence[Sequence[Hashable]]
    ranks: Sequence[float] | None = None
    partial_play: Sequence[Sequence[float]] | None = None
    match_weight: float = 1.0


@dataclass(frozen=True)
class BatchResult:
    """Output of ``rate_matches``.

    ratings : final rating of every player seen.
    probabilities : each match's pre-match outcome probability if requested
        (``None`` entries for matches with more than 12 teams), else ``None``.
    """

    ratings: dict[Hashable, Rating]
    probabilities: list[float | None] | None = None
