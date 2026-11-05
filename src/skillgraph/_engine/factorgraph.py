"""Factor-graph primitives for Gaussian expectation propagation.

TrueSkill 1 is one wiring of these; TrueSkill 2 adds factor types on top.
Each ``Variable`` keeps its marginal and the last message from every attached
factor; every message update returns the change in the marginal, which the
schedule (``run_chain``) uses to test for convergence.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from .._constants import DAMPING_SCHEDULE
from .gaussian import Gaussian


class Variable(Gaussian):
    """A Gaussian marginal plus the incoming message from each factor.

    The marginal is always recomputed as the exact product (sum of natural
    parameters) of the incoming messages, and a factor's cavity is the
    product of the *other* messages. Nothing is ever obtained by dividing
    one Gaussian by another: with large precisions that subtraction cancels
    catastrophically and can yield negative variances.
    """

    def __init__(self) -> None:
        super().__init__()
        self.messages: dict[Factor, Gaussian] = {}

    def cavity(self, factor: Factor) -> Gaussian:
        others = [m for f, m in self.messages.items() if f is not factor]
        return Gaussian(pi=math.fsum(m.pi for m in others), tau=math.fsum(m.tau for m in others))

    def _delta(self, other: Gaussian) -> float:
        """Scale-free change: mean shift in standard deviations, and relative
        change in standard deviation. (A raw natural-parameter difference is
        dominated by rounding noise when precisions are large.)"""
        if not (self.pi > 0 and other.pi > 0):
            return math.inf if self.pi != other.pi or self.tau != other.tau else 0.0
        if not (math.isfinite(self.pi) and math.isfinite(other.pi)):
            return math.inf
        d_mu = abs(self.mu - other.mu) * math.sqrt(min(self.pi, other.pi))
        d_sigma = abs(1.0 - math.sqrt(self.pi / other.pi))
        return max(d_mu, d_sigma)

    def update_message(self, factor: Factor, message: Gaussian) -> float:
        """Replace ``factor``'s message, recompute the marginal, return the change."""
        self.messages[factor] = message
        old = Gaussian(pi=self.pi, tau=self.tau)
        self.pi = math.fsum(m.pi for m in self.messages.values())
        self.tau = math.fsum(m.tau for m in self.messages.values())
        return self._delta(old)


class Factor:
    """Base class: registers a (uniform) message on each attached variable."""

    def __init__(self, variables: Sequence[Variable]) -> None:
        for var in variables:
            var.messages[self] = Gaussian()


class PriorFactor(Factor):
    """Prior N(mu, sigma^2 + drift^2) on a skill variable."""

    def __init__(self, var: Variable, prior: Gaussian, drift: float = 0.0) -> None:
        super().__init__([var])
        self.var = var
        self.prior = prior
        self.drift = drift

    def down(self) -> float:
        sigma = math.sqrt(self.prior.sigma**2 + self.drift**2)
        return self.var.update_message(self, Gaussian(self.prior.mu, sigma))


class LikelihoodFactor(Factor):
    """value ~ N(mean, variance): adds Gaussian noise between two variables."""

    def __init__(self, mean: Variable, value: Variable, variance: float) -> None:
        super().__init__([mean, value])
        self.mean = mean
        self.value = value
        self.variance = variance

    def _send(self, src: Variable, dst: Variable) -> float:
        msg = src.cavity(self)
        a = 1.0 / (1.0 + self.variance * msg.pi)
        return dst.update_message(self, Gaussian(pi=a * msg.pi, tau=a * msg.tau))

    def down(self) -> float:
        return self._send(self.mean, self.value)

    def up(self) -> float:
        return self._send(self.value, self.mean)


class SumFactor(Factor):
    """total = sum(coeff_i * term_i)."""

    def __init__(self, total: Variable, terms: Sequence[Variable], coeffs: Sequence[float]) -> None:
        if len(terms) != len(coeffs):
            raise ValueError("terms and coeffs must have the same length")
        super().__init__([total, *terms])
        self.total = total
        self.terms = list(terms)
        self.coeffs = [float(c) for c in coeffs]

    def down(self) -> float:
        return self._update(self.total, self.terms, self.coeffs)

    def up(self, index: int = 0) -> float:
        """Send a message to ``terms[index]`` by solving the sum for it."""
        pivot = self.coeffs[index]
        coeffs = [1.0 / pivot if i == index else -c / pivot for i, c in enumerate(self.coeffs)]
        sources = list(self.terms)
        sources[index] = self.total
        return self._update(self.terms[index], sources, coeffs)

    def _update(self, target: Variable, sources: Sequence[Variable], coeffs: Sequence[float]) -> float:
        cavities = [src.cavity(self) for src in sources]
        if any(c.pi == 0.0 for c in cavities):  # a source carries no information yet
            return target.update_message(self, Gaussian())
        mu = sum(coeff * c.mu for c, coeff in zip(cavities, coeffs))
        var = sum(coeff**2 / c.pi for c, coeff in zip(cavities, coeffs))
        pi = 1.0 / var
        return target.update_message(self, Gaussian(pi=pi, tau=pi * mu))


class TruncateFactor(Factor):
    """Conditions a difference variable on a win or a draw.

    ``moments(t, eps) -> (v, one_minus_w)`` gives the standardized mean shift
    and variance ratio (``truncation.win`` or ``truncation.draw``).
    """

    def __init__(
        self,
        var: Variable,
        moments: Callable[[float, float], tuple[float, float]],
        margin: float,
    ) -> None:
        super().__init__([var])
        self.var = var
        self.moments = moments
        self.margin = margin

    def up(self, damping: float = 1.0) -> float:
        """Send the moment-matched message. ``damping`` < 1 blends it with the
        previous message (in natural parameters) to stop EP oscillating."""
        cav = self.var.cavity(self)
        if not (cav.pi > 0 and math.isfinite(cav.pi) and math.isfinite(cav.tau)):
            raise FloatingPointError("invalid cavity distribution in TruncateFactor")
        sqrt_pi = math.sqrt(cav.pi)
        v, omw = self.moments(cav.tau / sqrt_pi, self.margin * sqrt_pi)
        # Message = (moment-matched marginal) / cavity, written out so that no
        # two large precisions are subtracted.
        w = 1.0 - omw
        msg = Gaussian(pi=cav.pi * w / omw, tau=(cav.tau * w + sqrt_pi * v) / omw)
        if damping < 1.0:
            old = self.var.messages[self]
            msg = Gaussian(
                pi=(1.0 - damping) * old.pi + damping * msg.pi,
                tau=(1.0 - damping) * old.tau + damping * msg.tau,
            )
        return self.var.update_message(self, msg)


def run_chain(
    diffs: Sequence[SumFactor],
    truncs: Sequence[TruncateFactor],
    min_delta: float,
    max_iterations: int,
) -> tuple[bool, float]:
    """Iterate EP over a chain of team differences until it converges.

    TrueSkill's schedule: sweep forward then backward over the difference
    factors. Starts undamped; if that does not converge within
    ``max_iterations`` sweeps, continues from the current state with each
    level of ``DAMPING_SCHEDULE`` in turn. Returns ``(converged, last_delta)``.
    """
    n = len(diffs)
    delta = math.inf
    for damping in DAMPING_SCHEDULE:
        for _ in range(max_iterations):
            delta = 0.0
            if n == 1:
                diffs[0].down()
                delta = truncs[0].up(damping)
            for k in range(n - 1):
                diffs[k].down()
                delta = max(delta, truncs[k].up(damping))
                diffs[k].up(1)
            for k in range(n - 1, 0, -1):
                diffs[k].down()
                delta = max(delta, truncs[k].up(damping))
                diffs[k].up(0)
            if delta <= min_delta * damping:
                return True, delta
    return False, delta
