from __future__ import annotations

"""Robust portfolio optimisation utilities."""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class RobustOptimizer:
    """Mean-variance optimiser with conservative covariance adjustment.

    The optimiser applies a worst-case adjustment to the covariance matrix by
    adding a scaled identity matrix.  This is equivalent to solving the
    portfolio optimisation problem over a Wasserstein ambiguity set where the
    adversary can inflate the covariance by ``ambiguity``.  Larger values of
    ``ambiguity`` therefore lead to more conservative (diversified) weights.
    """

    risk_aversion: float = 1.0
    ambiguity: float = 0.1
    weights: np.ndarray | None = field(default=None, init=False)

    def _worst_case_covariance(self, cov: np.ndarray) -> np.ndarray:
        """Return covariance inflated towards the identity matrix."""
        n = cov.shape[0]
        scale = np.trace(cov) / n if n else 1.0
        return cov + self.ambiguity * scale * np.eye(n)

    def compute_weights(
        self, expected_returns: Sequence[float], cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Return normalised weights under worst-case covariance assumptions.

        Parameters
        ----------
        expected_returns:
            Iterable of expected asset returns.
        cov_matrix:
            Empirical covariance matrix of asset returns.
        """
        mu = np.asarray(expected_returns, dtype=float)
        cov = np.asarray(cov_matrix, dtype=float)
        if cov.shape[0] != cov.shape[1]:
            raise ValueError("cov_matrix must be square")
        if mu.shape[0] != cov.shape[0]:
            raise ValueError("expected_returns and cov_matrix dimensions mismatch")
        robust_cov = self._worst_case_covariance(cov)
        inv = np.linalg.pinv(robust_cov)
        raw = inv @ mu / max(self.risk_aversion, 1e-12)
        weights = raw / np.sum(np.abs(raw))
        self.weights = weights
        return weights

    def diversification_ratio(self) -> float:
        """Return diversification metric based on the Herfindahl index."""
        if self.weights is None or len(self.weights) == 0:
            return 0.0
        return float((1.0 / np.sum(self.weights ** 2)) / len(self.weights))
