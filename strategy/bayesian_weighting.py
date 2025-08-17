from __future__ import annotations

"""Bayesian weighting of trading algorithms.

This module maintains posterior distributions of risk-adjusted returns for a
collection of algorithms.  Posteriors are modelled with a Normal-Inverse-Gamma
conjugate prior allowing closed-form updates from daily profit and loss (PnL)
observations.  Allocation weights are derived from the expected Sharpe ratio of
each algorithm's posterior, with a shrinkage factor that naturally favours
better-established (safer) strategies when evidence is weak.
"""

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np


@dataclass
class _NormalInverseGamma:
    """Track posterior parameters of a Normal-Inverse-Gamma distribution."""

    mu0: float
    lambda0: float
    alpha0: float
    beta0: float
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        """Update running statistics with a new observation ``x``."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    # Posterior parameters -------------------------------------------------
    @property
    def lambda_(self) -> float:
        return self.lambda0 + self.n

    @property
    def mu(self) -> float:
        if self.n == 0:
            return self.mu0
        return (self.lambda0 * self.mu0 + self.n * self.mean) / (self.lambda0 + self.n)

    @property
    def alpha(self) -> float:
        return self.alpha0 + self.n / 2

    @property
    def beta(self) -> float:
        if self.n == 0:
            return self.beta0
        return (
            self.beta0
            + 0.5 * self.m2
            + (self.lambda0 * self.n * (self.mean - self.mu0) ** 2)
            / (2 * (self.lambda0 + self.n))
        )


@dataclass
class BayesianWeighting:
    """Maintain Bayesian posteriors and derive allocation weights.

    Parameters
    ----------
    algorithms:
        Iterable of algorithm names to track.
    safety_strength:
        Pseudo-count controlling shrinkage toward safer strategies when data
        is scarce.  Higher values lead to more conservative allocations.
    prior_mean, prior_lambda, prior_alpha, prior_beta:
        Parameters of the Normal-Inverse-Gamma prior.
    """

    algorithms: Iterable[str]
    safety_strength: float = 10.0
    prior_mean: float = 0.0
    prior_lambda: float = 1.0
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    def __post_init__(self) -> None:
        self.posteriors: Dict[str, _NormalInverseGamma] = {
            name: _NormalInverseGamma(
                mu0=self.prior_mean,
                lambda0=self.prior_lambda,
                alpha0=self.prior_alpha,
                beta0=self.prior_beta,
            )
            for name in self.algorithms
        }

    # Logging --------------------------------------------------------------
    def log_pnl(self, algorithm: str, pnl: float) -> None:
        """Log a daily PnL observation for ``algorithm``."""
        if algorithm not in self.posteriors:
            raise KeyError(f"Unknown algorithm '{algorithm}'")
        self.posteriors[algorithm].update(pnl)

    # Inspection -----------------------------------------------------------
    def posterior(self, algorithm: str) -> Dict[str, float]:
        """Return a dictionary of posterior parameters for ``algorithm``."""
        p = self.posteriors[algorithm]
        return {
            "mu": p.mu,
            "lambda": p.lambda_,
            "alpha": p.alpha,
            "beta": p.beta,
            "n": p.n,
        }

    # Weighting ------------------------------------------------------------
    def _expected_sharpe(self, p: _NormalInverseGamma) -> float:
        if p.n == 0:
            return 0.0
        variance = p.beta / (p.alpha - 1) if p.alpha > 1 else np.inf
        sharpe = p.mu / np.sqrt(variance + 1e-9)
        shrink = p.n / (p.n + self.safety_strength)
        return sharpe * shrink

    def weights(self, sample: bool = False) -> Dict[str, float]:
        """Return allocation weights derived from the posterior.

        If ``sample`` is True, Monte Carlo samples of the posteriors are used
        instead of the expected values which can encourage additional
        exploration.
        """

        scores = []
        names = list(self.posteriors)
        for name in names:
            p = self.posteriors[name]
            if sample and p.n > 0:
                tau = np.random.gamma(p.alpha, 1.0 / p.beta)
                sigma = np.sqrt(1.0 / tau)
                mu = np.random.normal(p.mu, np.sqrt(1.0 / (p.lambda_ * tau)))
                sharpe = mu / (sigma + 1e-9)
                shrink = p.n / (p.n + self.safety_strength)
                score = sharpe * shrink
            else:
                score = self._expected_sharpe(p)
            scores.append(max(0.0, score))

        scores_arr = np.asarray(scores)
        if not np.any(scores_arr):
            scores_arr = np.ones_like(scores_arr)
        weights = scores_arr / scores_arr.sum()
        return dict(zip(names, weights))


__all__ = ["BayesianWeighting"]
