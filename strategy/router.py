from __future__ import annotations

"""Contextual bandit based strategy router.

This module implements a lightweight LinUCB contextual bandit that chooses
between different trading algorithms based on regime features such as market
volatility, trend strength and a numeric market regime indicator.  After each
trade the realised profit and loss is fed back to the bandit so allocation
gradually shifts toward the best-performing algorithm in the current regime.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

FeatureDict = Dict[str, float]
Algorithm = Callable[[FeatureDict], float]


def _feature_vector(features: FeatureDict) -> np.ndarray:
    """Return a 3x1 feature vector of volatility, trend strength and regime."""
    return np.array(
        [
            features.get("volatility", 0.0),
            features.get("trend_strength", 0.0),
            features.get("regime", 0.0),
        ]
    ).reshape(-1, 1)


@dataclass
class StrategyRouter:
    """Route orders between algorithms using a LinUCB contextual bandit."""

    algorithms: Dict[str, Algorithm] = field(default_factory=dict)
    alpha: float = 0.1

    def __post_init__(self) -> None:
        if not self.algorithms:
            # Default placeholder algorithms.  They simply return a constant
            # action; real applications should supply concrete implementations.
            self.algorithms = {
                "mean_reversion": lambda _: -1.0,
                "trend_following": lambda _: 1.0,
                "rl_policy": lambda _: 0.0,
            }
        # include regime dimension alongside volatility and trend strength
        self.dim = 3
        self.A: Dict[str, np.ndarray] = {
            name: np.identity(self.dim) for name in self.algorithms
        }
        self.b: Dict[str, np.ndarray] = {
            name: np.zeros((self.dim, 1)) for name in self.algorithms
        }
        self.history: List[Tuple[FeatureDict, float, str]] = []

    # Registration -----------------------------------------------------
    def register(self, name: str, algorithm: Algorithm) -> None:
        """Register a new algorithm with fresh bandit parameters."""
        self.algorithms[name] = algorithm
        self.A[name] = np.identity(self.dim)
        self.b[name] = np.zeros((self.dim, 1))

    # Selection --------------------------------------------------------
    def select(self, features: FeatureDict) -> str:
        """Return the algorithm name with the highest UCB score."""
        x = _feature_vector(features)
        best_name = None
        best_score = -np.inf
        for name in self.algorithms:
            A_inv = np.linalg.inv(self.A[name])
            theta = A_inv @ self.b[name]
            mean = float((theta.T @ x).item())
            bonus = float(self.alpha * np.sqrt((x.T @ A_inv @ x).item()))
            score = mean + bonus
            if score > best_score:
                best_score = score
                best_name = name
        assert best_name is not None  # for type checkers
        return best_name

    def act(self, features: FeatureDict) -> Tuple[str, float]:
        """Select an algorithm and return its action."""
        name = self.select(features)
        action = self.algorithms[name](features)
        return name, action

    # Learning ---------------------------------------------------------
    def update(self, features: FeatureDict, reward: float, algorithm: str) -> None:
        """Update bandit parameters with observed ``reward``."""
        x = _feature_vector(features)
        self.A[algorithm] += x @ x.T
        self.b[algorithm] += reward * x
        self.history.append((features, reward, algorithm))

    # Convenience ------------------------------------------------------
    def log_reward(self, features: FeatureDict, reward: float, algorithm: str) -> None:
        """Alias for :meth:`update` for semantic clarity."""
        self.update(features, reward, algorithm)


__all__ = ["StrategyRouter"]
