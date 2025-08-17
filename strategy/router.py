from __future__ import annotations

"""Contextual bandit based strategy router.

This module implements a lightweight LinUCB contextual bandit that chooses
between different trading algorithms based on regime features such as market
volatility, trend strength and a numeric market regime indicator.  After each
trade the realised profit and loss is fed back to the bandit so allocation
gradually shifts toward the best-performing algorithm in the current regime.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

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
    scoreboard_path: Path | str = Path("reports/scoreboard.parquet")

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
        self.scoreboard_path = Path(self.scoreboard_path)
        self.scoreboard = self._load_scoreboard()

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
        regime = features.get("regime")
        best_name = None
        best_score = -np.inf
        for name in self.algorithms:
            A_inv = np.linalg.inv(self.A[name])
            theta = A_inv @ self.b[name]
            mean = float((theta.T @ x).item())
            bonus = float(self.alpha * np.sqrt((x.T @ A_inv @ x).item()))
            score = mean + bonus + self._scoreboard_weight(regime, name)
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
        self._update_scoreboard(features.get("regime"), algorithm)

    # Convenience ------------------------------------------------------
    def log_reward(self, features: FeatureDict, reward: float, algorithm: str) -> None:
        """Alias for :meth:`update` for semantic clarity."""
        self.update(features, reward, algorithm)

    # Scoreboard -------------------------------------------------------
    def _load_scoreboard(self) -> pd.DataFrame:
        if self.scoreboard_path.exists():
            return pd.read_parquet(self.scoreboard_path)
        return pd.DataFrame(columns=["sharpe", "drawdown"], index=pd.MultiIndex.from_tuples([], names=["regime", "algorithm"]))

    def _scoreboard_weight(self, regime: float | None, algorithm: str) -> float:
        if regime is None or self.scoreboard.empty:
            return 0.0
        try:
            return float(self.scoreboard.loc[(regime, algorithm), "sharpe"])
        except KeyError:
            return 0.0

    def _update_scoreboard(self, regime: float | None, algorithm: str) -> None:
        if regime is None:
            return
        rewards = [r for f, r, a in self.history if a == algorithm and f.get("regime") == regime]
        if len(rewards) < 2:
            return
        arr = np.asarray(rewards)
        sharpe = float(arr.mean() / (arr.std() + 1e-9))
        cumulative = (1 + arr).cumprod()
        drawdown = float((np.maximum.accumulate(cumulative) - cumulative).max())
        self.scoreboard.loc[(regime, algorithm), ["sharpe", "drawdown"]] = [sharpe, drawdown]
        self.scoreboard_path.parent.mkdir(parents=True, exist_ok=True)
        self.scoreboard.to_parquet(self.scoreboard_path)


__all__ = ["StrategyRouter"]
