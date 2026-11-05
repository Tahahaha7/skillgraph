"""Gaussian distributions in natural parameters.

Precision ``pi = 1/sigma^2`` and precision-adjusted mean ``tau = mu * pi``:
combining messages in expectation propagation is then just adding them.
"""

from __future__ import annotations

import math


class Gaussian:
    """A (possibly improper) Gaussian; ``Gaussian()`` is the uniform pi = tau = 0."""

    __slots__ = ("pi", "tau")

    def __init__(
        self,
        mu: float | None = None,
        sigma: float | None = None,
        *,
        pi: float = 0.0,
        tau: float = 0.0,
    ) -> None:
        if mu is not None:
            if sigma is None or not sigma > 0:
                raise ValueError("a positive sigma is required when mu is given")
            pi = sigma**-2
            tau = pi * mu
        self.pi = float(pi)
        self.tau = float(tau)

    @property
    def mu(self) -> float:
        return self.tau / self.pi if self.pi else 0.0

    @property
    def sigma(self) -> float:
        return math.sqrt(1.0 / self.pi) if self.pi else math.inf

    def __repr__(self) -> str:
        return f"Gaussian(mu={self.mu:.3f}, sigma={self.sigma:.3f})"
