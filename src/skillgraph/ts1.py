"""TrueSkill 1 (Herbrich, Minka & Graepel, NIPS 2006).

Two inference paths with identical results (enforced by the tests): two teams
use the exact closed form; three or more use EP on the factor graph.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

from scipy.special import ndtri

from ._constants import EXACT_LEAVE_ONE_OUT_MAX
from ._engine.factorgraph import (
    LikelihoodFactor,
    PriorFactor,
    SumFactor,
    TruncateFactor,
    Variable,
    run_chain,
)
from ._engine.gaussian import Gaussian
from ._engine.truncation import draw, win
from ._model import PartialPlay, SkillModel, class_or_instance, positive_finite
from ._types import DEFAULT_MU, DEFAULT_SIGMA, ConvergenceWarning, Rating, Team


class TS1(SkillModel):
    """TrueSkill 1.

    Call on the class for default settings, or instantiate to customize::

        TS1.rate(teams, ranks)
        TS1(beta=4.0).rate(teams, ranks)

    mu, sigma : prior mean and std for new players.
    beta : performance noise std (default sigma/2).
    tau : skill drift std added before each match (default sigma/100).
    draw_probability : chance of a draw between equal teams; set it to your
        observed draw rate (under the default 0.10 a draw is surprising and
        moves ratings a lot).
    min_delta : EP convergence threshold (3+ teams): largest change in a
        difference's mean, in standard deviations, or in its relative sd.
    max_iterations : EP sweeps per damping level before trying more damping.
    """

    def __init__(
        self,
        mu: float = DEFAULT_MU,
        sigma: float = DEFAULT_SIGMA,
        beta: float | None = None,
        tau: float | None = None,
        draw_probability: float = 0.10,
        min_delta: float = 1e-4,
        max_iterations: int = 100,
    ) -> None:
        self.mu = float(mu)
        if not math.isfinite(self.mu):
            raise ValueError("mu must be finite")
        self.sigma = positive_finite("sigma", sigma)
        self.beta = positive_finite("beta", self.sigma / 2 if beta is None else beta)
        self.tau = positive_finite("tau", self.sigma / 100 if tau is None else tau, allow_zero=True)
        self.draw_probability = float(draw_probability)
        if not 0.0 <= self.draw_probability < 1.0:
            raise ValueError("draw_probability must be in [0, 1)")
        self.min_delta = positive_finite("min_delta", min_delta)
        self.max_iterations = int(max_iterations)
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")

    def __repr__(self) -> str:
        return (
            f"TS1(mu={self.mu}, sigma={self.sigma}, beta={self.beta}, tau={self.tau}, "
            f"draw_probability={self.draw_probability})"
        )

    @class_or_instance
    def create_rating(self, mu: float | None = None, sigma: float | None = None) -> Rating:
        """A new player's rating (the model's prior unless overridden)."""
        return Rating(self.mu if mu is None else mu, self.sigma if sigma is None else sigma)

    @class_or_instance
    def draw_margin(self, n_players: int) -> float:
        """Performance difference below which a match between n_players is a draw."""
        return float(ndtri((self.draw_probability + 1.0) / 2.0)) * math.sqrt(n_players) * self.beta

    @class_or_instance
    def rate(
        self,
        teams: Sequence[Team],
        ranks: Sequence[float] | None = None,
        partial_play: PartialPlay | None = None,
        match_weight: float = 1.0,
    ) -> list[list[Rating]]:
        """Update ratings after one match.

        ranks : lower is better; equal ranks are draws (default: input order).
        partial_play : fraction of the match each player took part in, in (0, 1].
        match_weight : how much this match counts, in [0, 1]; scales its
            evidence (0 = only skill drift applies).
        Returns new ratings in the input order.
        """
        teams_, pp = self._check_teams(teams, partial_play)
        n = len(teams_)
        ranks_ = self._check_ranks(n, ranks)
        alpha = self._check_match_weight(match_weight)
        if self.draw_probability == 0.0 and len(set(ranks_)) < n:
            raise ValueError("a draw was reported but draw_probability is 0")
        order = sorted(range(n), key=ranks_.__getitem__)  # stable: ties keep input order
        update = self._rate_two if n == 2 else self._rate_graph
        new = update([teams_[i] for i in order], [ranks_[i] for i in order], [pp[i] for i in order], alpha)
        out: list[list[Rating]] = [[] for _ in range(n)]
        for pos, i in enumerate(order):
            out[i] = new[pos]
        return out

    def _team_moments(
        self, teams: list[list[Rating]], partial_play: list[list[float]]
    ) -> tuple[list[float], list[float]]:
        extra = self.tau**2 + self.beta**2
        means = [math.fsum(w * r.mu for r, w in zip(t, ws)) for t, ws in zip(teams, partial_play)]
        variances = [
            math.fsum(w * w * (r.sigma**2 + extra) for r, w in zip(t, ws)) for t, ws in zip(teams, partial_play)
        ]
        return means, variances

    # Both update paths take teams sorted by rank and return ratings in that order.

    def _rate_two(
        self, teams: list[list[Rating]], ranks: list[float], partial_play: list[list[float]], alpha: float
    ) -> list[list[Rating]]:
        """Exact closed form for two teams, in natural parameters (no cancellation)."""
        beta2, tau2 = self.beta**2, self.tau**2
        players = [(r, w, sign) for t, ws, sign in zip(teams, partial_play, (1.0, -1.0)) for r, w in zip(t, ws)]
        s2 = [r.sigma**2 + tau2 for r, _, _ in players]
        q = [w * w * (v + beta2) for (_, w, _), v in zip(players, s2)]
        c2 = math.fsum(q)
        c = math.sqrt(c2)
        m_d = math.fsum(sign * w * r.mu for r, w, sign in players)
        moments = draw if ranks[0] == ranks[1] else win
        v, omw = moments(m_d / c, self.draw_margin(len(teams[0]) + len(teams[1])) / c)
        # Message onto the difference d: precision pi_d, and its natural mean
        # relative to d's prior mean, eta_d - pi_d * m_d = v / (c * omw).
        pi_d = (1.0 - omw) / (omw * c2)
        shift_d = v / (c * omw)
        exact = len(players) <= EXACT_LEAVE_ONE_OUT_MAX
        new = []
        for i, ((r, w, sign), var_i) in enumerate(zip(players, s2)):
            others = math.fsum(q[:i] + q[i + 1 :]) if exact else c2 - q[i]  # variance of d minus player i
            a = sign * w
            denom = 1.0 + (others + w * w * beta2) * pi_d
            pi_post = 1.0 / var_i + alpha * a * a * pi_d / denom
            mu_post = r.mu + alpha * a * shift_d / (denom * pi_post)  # prior + shift: exact when no evidence
            new.append(Rating(mu_post, math.sqrt(1.0 / pi_post)))
        k = len(teams[0])
        return [new[:k], new[k:]]

    def _rate_graph(
        self, teams: list[list[Rating]], ranks: list[float], partial_play: list[list[float]], alpha: float
    ) -> list[list[Rating]]:
        """EP on the factor graph: skill -> performance -> team -> difference -> outcome."""
        skills = [[Variable() for _ in t] for t in teams]
        perfs = [[Variable() for _ in t] for t in teams]
        team_vars = [Variable() for _ in teams]
        diff_vars = [Variable() for _ in teams[1:]]

        prior_fs = [
            [PriorFactor(s, Gaussian(r.mu, r.sigma), self.tau) for s, r in zip(ss, t)] for ss, t in zip(skills, teams)
        ]
        perf_fs = [[LikelihoodFactor(s, p, self.beta**2) for s, p in zip(ss, ps)] for ss, ps in zip(skills, perfs)]
        team_fs = [SumFactor(tv, ps, w) for tv, ps, w in zip(team_vars, perfs, partial_play)]
        diff_fs = [SumFactor(d, team_vars[k : k + 2], [1.0, -1.0]) for k, d in enumerate(diff_vars)]
        trunc_fs = [
            TruncateFactor(
                d,
                draw if ranks[k] == ranks[k + 1] else win,
                self.draw_margin(len(teams[k]) + len(teams[k + 1])),
            )
            for k, d in enumerate(diff_vars)
        ]

        for row in prior_fs:
            for f in row:
                f.down()
        for row in perf_fs:
            for lf in row:
                lf.down()
        for sf in team_fs:
            sf.down()
        converged, delta = run_chain(diff_fs, trunc_fs, self.min_delta, self.max_iterations)
        if not converged:
            warnings.warn(
                f"message passing did not converge (last delta {delta:.3g} > min_delta "
                f"{self.min_delta:.3g}, even with damping)",
                ConvergenceWarning,
                stacklevel=3,
            )
        diff_fs[0].up(0)
        diff_fs[-1].up(1)
        for sf in team_fs:
            for i in range(len(sf.terms)):
                sf.up(i)
        for row in perf_fs:
            for lf in row:
                lf.up()

        out = []
        for ss, pfs, lfs in zip(skills, prior_fs, perf_fs):
            row_out = []
            for s, prior_f, perf_f in zip(ss, pfs, lfs):
                prior = s.messages[prior_f]
                evidence = s.messages[perf_f]  # everything the match taught us about this player
                pi_post = prior.pi + alpha * evidence.pi
                mu_post = prior.mu + alpha * (evidence.tau - evidence.pi * prior.mu) / pi_post
                if not (pi_post > 0 and math.isfinite(pi_post) and math.isfinite(mu_post)):
                    raise FloatingPointError("rating update produced an invalid distribution")
                row_out.append(Rating(mu_post, math.sqrt(1.0 / pi_post)))
            out.append(row_out)
        return out
