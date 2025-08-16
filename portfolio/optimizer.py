"""Portfolio optimizer that computes asset weights and handles rebalancing."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Sequence

import numpy as np

from analytics.metrics_store import record_metric

logger = logging.getLogger(__name__)


@dataclass
class PortfolioOptimizer:
    """Mean-variance optimizer for computing portfolio weights."""

    risk_aversion: float = 1.0
    weights: np.ndarray | None = field(default=None, init=False)

    def compute_weights(self, expected_returns: Sequence[float], cov_matrix: np.ndarray) -> np.ndarray:
        """Return normalized weights that maximize mean-variance utility.

        Parameters
        ----------
        expected_returns:
            Iterable of expected asset returns.
        cov_matrix:
            Covariance matrix of asset returns.
        """
        mu = np.asarray(expected_returns, dtype=float)
        cov = np.asarray(cov_matrix, dtype=float)
        if cov.shape[0] != cov.shape[1]:
            raise ValueError("cov_matrix must be square")
        if mu.shape[0] != cov.shape[0]:
            raise ValueError("expected_returns and cov_matrix dimensions mismatch")
        inv = np.linalg.pinv(cov)
        raw = inv @ mu / max(self.risk_aversion, 1e-12)
        weights = raw / np.sum(np.abs(raw))
        self.weights = weights
        return weights

    def diversification_ratio(self) -> float:
        """Return a simple diversification metric based on Herfindahl index."""
        if self.weights is None or len(self.weights) == 0:
            return 0.0
        return float((1.0 / np.sum(self.weights ** 2)) / len(self.weights))


@dataclass
class PortfolioRebalancer:
    """Periodically rebalance portfolio and log drawdown/diversification."""

    optimizer: PortfolioOptimizer
    rebalance_interval: int = 20
    step: int = 0
    returns: list[float] = field(default_factory=list)

    def update(
        self,
        expected_returns: Sequence[float],
        cov_matrix: np.ndarray,
        portfolio_return: float | None = None,
    ) -> np.ndarray | None:
        """Update portfolio state and rebalance when interval reached.

        ``portfolio_return`` should be the realized return since the last call.
        When ``step`` reaches ``rebalance_interval`` the optimizer is invoked to
        compute new weights which are then logged along with drawdown and
        diversification metrics.
        """
        if portfolio_return is not None:
            self.returns.append(float(portfolio_return))
        self.step += 1
        if self.step % self.rebalance_interval == 0:
            weights = self.optimizer.compute_weights(expected_returns, cov_matrix)
            drawdown = self._compute_drawdown()
            try:
                record_metric("portfolio_drawdown", drawdown)
                record_metric("diversification_ratio", self.optimizer.diversification_ratio())
            except Exception:
                pass
            logger.info(
                "Rebalanced portfolio: drawdown=%.4f diversification=%.4f",
                drawdown,
                self.optimizer.diversification_ratio(),
            )
            return weights
        return self.optimizer.weights

    def _compute_drawdown(self) -> float:
        if not self.returns:
            return 0.0
        cumulative = np.cumprod([1 + r for r in self.returns])
        peak = np.maximum.accumulate(cumulative)
        dd = (cumulative[-1] - peak[-1]) / peak[-1]
        return float(dd)
