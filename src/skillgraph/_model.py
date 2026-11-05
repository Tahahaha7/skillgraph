"""Behaviour shared by every rating model.

A model subclass implements ``rate``, ``draw_margin``, ``create_rating`` and
``_team_moments`` and sets ``beta``. It inherits the two call styles,
input validation, ``rate_1vs1``, ``rate_matches`` and all predictions.
"""

from __future__ import annotations

import functools
import math
import types
from collections.abc import Hashable, Iterable, Sequence
from typing import Any, Callable, ClassVar

import numpy as np
from scipy.special import log_ndtr, ndtr

from ._constants import EXACT_WIN_PROB_MAX_TEAMS, MC_SAMPLES, OUTCOME_PROB_MAX_TEAMS
from ._engine.probability import best_probability, chain_box_probability, log_box, monte_carlo_best
from ._types import BatchResult, Match, Rating, Team

PartialPlay = Sequence[Sequence[float]]


class class_or_instance:  # lower-case: used as a decorator
    """Make a method callable on the class (shared default model) or an instance.

    ``TS1.rate(...)`` means ``TS1._default().rate(...)``; ``TS1(beta=4).rate(...)``
    uses that instance. Safe because models never mutate their settings.
    """

    def __init__(self, func: Callable[..., Any]) -> None:
        self.func = func
        functools.update_wrapper(self, func)

    def __get__(self, obj: Any, cls: type) -> Callable[..., Any]:
        return types.MethodType(self.func, cls._default() if obj is None else obj)  # type: ignore[attr-defined]


def positive_finite(name: str, x: float, allow_zero: bool = False) -> float:
    """Validate a model setting."""
    x = float(x)
    if not (math.isfinite(x) and (x >= 0 if allow_zero else x > 0)):
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'} and finite")
    return x


class SkillModel:
    """Base class for rating models (see module docstring)."""

    beta: float
    _default_instance: ClassVar[SkillModel | None] = None

    @classmethod
    def _default(cls) -> SkillModel:
        """The shared default-settings model behind class-level calls (one per subclass)."""
        inst = cls.__dict__.get("_default_instance")
        if inst is None:
            inst = cls()
            cls._default_instance = inst
        return inst

    # ----------------------------------------------------- implemented by models
    def rate(
        self,
        teams: Sequence[Team],
        ranks: Sequence[float] | None = None,
        partial_play: PartialPlay | None = None,
        match_weight: float = 1.0,
    ) -> list[list[Rating]]:
        raise NotImplementedError

    def draw_margin(self, n_players: int) -> float:
        raise NotImplementedError

    def create_rating(self) -> Rating:
        raise NotImplementedError

    def _team_moments(
        self, teams: list[list[Rating]], partial_play: list[list[float]]
    ) -> tuple[list[float], list[float]]:
        """Predictive mean and variance of each team's performance at match time."""
        raise NotImplementedError

    # --------------------------------------------------------------- validation
    @staticmethod
    def _check_teams(
        teams: Sequence[Team], partial_play: PartialPlay | None
    ) -> tuple[list[list[Rating]], list[list[float]]]:
        teams_ = [list(t) for t in teams]
        if len(teams_) < 2:
            raise ValueError("need at least two teams")
        for t in teams_:
            if not t:
                raise ValueError("teams must be non-empty")
            for r in t:
                if not isinstance(r, Rating):
                    raise TypeError(f"expected Rating, got {type(r).__name__}")
        if partial_play is None:
            return teams_, [[1.0] * len(t) for t in teams_]
        pp = [[float(x) for x in row] for row in partial_play]
        if len(pp) != len(teams_) or any(len(p) != len(t) for p, t in zip(pp, teams_)):
            raise ValueError("partial_play must match the shape of teams")
        if any(not (math.isfinite(x) and 0.0 < x <= 1.0) for row in pp for x in row):
            raise ValueError("partial_play values must be in (0, 1]")
        return teams_, pp

    @staticmethod
    def _check_ranks(n: int, ranks: Sequence[float] | None) -> list[float]:
        out = [float(i) for i in range(n)] if ranks is None else [float(r) for r in ranks]
        if len(out) != n:
            raise ValueError("ranks must have one entry per team")
        if not all(math.isfinite(r) for r in out):
            raise ValueError("ranks must be finite")
        return out

    @staticmethod
    def _check_match_weight(match_weight: float) -> float:
        x = float(match_weight)
        if not (math.isfinite(x) and 0.0 <= x <= 1.0):
            raise ValueError("match_weight must be in [0, 1]")
        return x

    # ------------------------------------------------------------------ rating
    @class_or_instance
    def rate_1vs1(self, winner: Rating, loser: Rating, drawn: bool = False) -> tuple[Rating, Rating]:
        """``rate`` for two single players; the first one won unless ``drawn``."""
        (new_winner,), (new_loser,) = self.rate([[winner], [loser]], ranks=[0, 0] if drawn else [0, 1])
        return new_winner, new_loser

    @class_or_instance
    def rate_matches(
        self,
        matches: Iterable[Match],
        ratings: dict[Hashable, Rating] | None = None,
        *,
        with_probabilities: bool = False,
    ) -> BatchResult:
        """Process matches in order, tracking ratings by player id.

        ratings : starting ratings (not modified); unseen ids start at ``create_rating()``.
        with_probabilities : also record each match's pre-match outcome
            probability (e.g. for log-loss).
        """
        current = {} if ratings is None else dict(ratings)
        probs: list[float | None] | None = [] if with_probabilities else None
        for idx, m in enumerate(matches):
            ids = [pid for team in m.teams for pid in team]
            if len(ids) != len(set(ids)):
                raise ValueError(f"match {idx}: a player appears more than once")
            teams = [[current[p] if p in current else self.create_rating() for p in team] for team in m.teams]
            if probs is not None:
                fits = len(teams) <= OUTCOME_PROB_MAX_TEAMS
                probs.append(self.outcome_probability(teams, m.ranks, m.partial_play) if fits else None)
            for team_ids, team_new in zip(m.teams, self.rate(teams, m.ranks, m.partial_play, m.match_weight)):
                current.update(zip(team_ids, team_new))
        return BatchResult(current, probs)

    # ------------------------------------------------------------- predictions
    @class_or_instance
    def quality(self, teams: Sequence[Team], partial_play: PartialPlay | None = None) -> float:
        """Match quality in [0, 1]: draw probability relative to a perfectly even match.

        Classic TrueSkill definition (skill sigma and beta; no drift term).
        """
        teams_, pp = self._check_teams(teams, partial_play)
        mus = np.array([r.mu for t in teams_ for r in t])
        var = np.array([r.sigma**2 for t in teams_ for r in t])
        w = np.array([x for row in pp for x in row])
        a = np.zeros((len(mus), len(teams_) - 1))  # player -> consecutive team differences
        start = np.cumsum([0] + [len(t) for t in teams_])
        for k in range(len(teams_) - 1):
            a[start[k] : start[k + 1], k] = w[start[k] : start[k + 1]]
            a[start[k + 1] : start[k + 2], k] = -w[start[k + 1] : start[k + 2]]
        ata = self.beta**2 * a.T @ a
        middle = ata + a.T @ (var[:, None] * a)
        v = mus @ a
        exponent = -0.5 * float(v @ np.linalg.solve(middle, v))
        # log-determinants: plain determinants overflow with many teams
        return math.exp(exponent + 0.5 * (np.linalg.slogdet(ata)[1] - np.linalg.slogdet(middle)[1]))

    @class_or_instance
    def win_probabilities(
        self,
        teams: Sequence[Team],
        partial_play: PartialPlay | None = None,
        *,
        n_samples: int = MC_SAMPLES,
        seed: int | None = 0,
    ) -> list[float]:
        """P(each team has the best performance). Draws are ignored; sums to 1.

        Exact (~1e-15) up to 25 teams; seeded Monte Carlo above
        (standard error <= 0.5/sqrt(n_samples)).
        """
        teams_, pp = self._check_teams(teams, partial_play)
        means, variances = self._team_moments(teams_, pp)
        m, sd = np.array(means), np.sqrt(np.array(variances))
        if len(teams_) == 2:
            p = float(ndtr((m[0] - m[1]) / math.hypot(sd[0], sd[1])))
            return [p, 1.0 - p]
        if len(teams_) <= EXACT_WIN_PROB_MAX_TEAMS:
            probs = [best_probability(i, m, sd) for i in range(len(teams_))]
            total = math.fsum(probs)
            return [p / total for p in probs]
        return monte_carlo_best(m, sd, n_samples, seed)

    @class_or_instance
    def outcome_probability(
        self,
        teams: Sequence[Team],
        ranks: Sequence[float] | None = None,
        partial_play: PartialPlay | None = None,
        *,
        log: bool = False,
    ) -> float:
        """Model probability of the full result (ranks, including draws).

        The evidence ``rate`` conditions on, so the right quantity for
        log-loss. Accuracy: 2 teams exact and log-stable for any upset.
        3+ teams: absolute error ~1e-15 (3 teams) or ~1e-7 (4-12 teams,
        deterministic, ~50 ms per call); outcomes rarer than that may come
        back as 0 (log -inf), so FFA log-loss on very unlikely results is
        unreliable. Raises ValueError above 12 teams.
        """
        teams_, pp = self._check_teams(teams, partial_play)
        n = len(teams_)
        if n > OUTCOME_PROB_MAX_TEAMS:
            raise ValueError(
                f"outcome_probability supports up to {OUTCOME_PROB_MAX_TEAMS} teams; "
                "use win_probabilities for big FFAs"
            )
        ranks_ = self._check_ranks(n, ranks)
        order = sorted(range(n), key=ranks_.__getitem__)
        means, variances = self._team_moments(teams_, pp)
        margins = [self.draw_margin(len(teams_[a]) + len(teams_[b])) for a, b in zip(order, order[1:])]
        drawn = [ranks_[a] == ranks_[b] for a, b in zip(order, order[1:])]
        if n == 2:
            a, b = order
            c = math.sqrt(variances[a] + variances[b])
            d, eps = means[a] - means[b], margins[0]
            lp = log_box(-(eps + d) / c, (eps - d) / c) if drawn[0] else float(log_ndtr((d - eps) / c))
            return lp if log else math.exp(lp)
        lower = np.array([-e if dr else e for e, dr in zip(margins, drawn)])
        upper = np.array([e if dr else np.inf for e, dr in zip(margins, drawn)])
        p = chain_box_probability([means[i] for i in order], [variances[i] for i in order], lower, upper)
        if log:
            return math.log(p) if p > 0 else -math.inf
        return p
