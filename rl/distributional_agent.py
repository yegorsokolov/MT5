"""Simple distributional reinforcement learning agents."""

from __future__ import annotations

from typing import List
import math


class DistributionalAgent:
    """Quantile-based agent maintaining return distributions per action.

    The agent stores observed rewards for each action and derives risk metrics
    from the empirical distribution.  While extremely lightweight, this mirrors
    the intuition behind algorithms such as C51 or QR-DQN where the policy
    reasons about the full return distribution instead of only its expectation.
    """

    def __init__(self, n_actions: int, n_quantiles: int = 51) -> None:
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.returns: List[List[float]] = [[] for _ in range(n_actions)]

    def act(self, obs) -> int:  # pragma: no cover - trivial action selection
        """Select the action with the highest expected return."""
        means = [sum(r) / len(r) if r else 0.0 for r in self.returns]
        return max(range(self.n_actions), key=lambda a: means[a])

    def update(self, action: int, reward: float) -> None:
        """Update quantiles for ``action`` towards ``reward``.

        Parameters
        ----------
        action: int
            Index of the action that was taken.
        reward: float
            Observed return used for the quantile regression update.
        """

        self.returns[action].append(reward)

    # Risk metrics ---------------------------------------------------------
    def value_at_risk(self, action: int, alpha: float) -> float:
        """Return the alpha-level Value at Risk estimate for ``action``."""
        history = sorted(self.returns[action])
        if not history:
            return 0.0
        idx = min(len(history) - 1, max(0, int(alpha * len(history))))
        return history[idx]

    def sharpe_ratio(self, action: int) -> float:
        """Estimate the Sharpe ratio from the quantile distribution."""
        qs = self.returns[action]
        if not qs:
            return 0.0
        mean = sum(qs) / len(qs)
        var = sum((q - mean) ** 2 for q in qs) / len(qs)
        std = math.sqrt(var) if var > 0 else 0.0
        return mean / std if std > 0 else 0.0


class MeanAgent:
    """Baseline agent tracking only mean returns.

    This represents a standard value-based agent that ignores the return
    distribution.  Risk metrics are therefore crude approximations.
    """

    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions
        self.means = [0.0 for _ in range(n_actions)]
        self.counts = [0 for _ in range(n_actions)]

    def act(self, obs) -> int:  # pragma: no cover - trivial action selection
        return max(range(self.n_actions), key=lambda a: self.means[a])

    def update(self, action: int, reward: float) -> None:
        self.counts[action] += 1
        n = self.counts[action]
        m = self.means[action]
        self.means[action] = m + (reward - m) / n

    def value_at_risk(self, action: int, alpha: float) -> float:
        # With no distributional information we fall back to the mean estimate.
        return self.means[action]

    def sharpe_ratio(self, action: int) -> float:
        # Assume unit variance leading to a risk-unadjusted return.
        return self.means[action]


__all__ = ["DistributionalAgent", "MeanAgent"]
