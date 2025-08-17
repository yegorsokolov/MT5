from __future__ import annotations

"""Lightweight deterministic world model for model-based RL."""

from dataclasses import dataclass
from typing import Iterable, Sequence

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - fallback if numpy not available
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import gym
    from gym import spaces
except Exception:  # pragma: no cover - allow running without gym
    gym = None  # type: ignore
    spaces = None  # type: ignore


@dataclass
class Transition:
    """Single transition used to train the world model."""

    state: Sequence[float]
    action: Sequence[float]
    next_state: Sequence[float]
    reward: float


class WorldModel:
    """Simple linear world model.

    The model predicts next state and reward from the current state and action.
    A small quadratic feature basis is used which is sufficient for the unit
    tests.  Parameters are fitted with a least squares solution.
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._W: np.ndarray | None = None
        self._b: np.ndarray | None = None

    # internal -----------------------------------------------------------------
    def _features(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Construct feature matrix including quadratic action terms."""
        if np is None:  # pragma: no cover - numpy required for features
            raise RuntimeError("numpy is required for WorldModel")
        s = np.atleast_2d(state)
        a = np.atleast_2d(action)
        feats = np.concatenate([s, a, a**2], axis=-1)
        return np.concatenate([feats, np.ones((feats.shape[0], 1))], axis=-1)

    # public API ---------------------------------------------------------------
    def fit(
        self,
        states: Sequence[Sequence[float]],
        actions: Sequence[Sequence[float]],
        next_states: Sequence[Sequence[float]],
        rewards: Sequence[float],
    ) -> None:
        """Fit model parameters from arrays."""
        if np is None:  # pragma: no cover - numpy required for training
            raise RuntimeError("numpy is required for WorldModel")
        X = self._features(np.array(states), np.array(actions))
        Y = np.concatenate(
            [np.array(next_states), np.array(rewards).reshape(-1, 1)], axis=-1
        )
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        self._W = beta[:-1]
        self._b = beta[-1]

    def train(self, transitions: Iterable[Transition]) -> None:
        """Convenience wrapper to train from an iterable of ``Transition``."""
        states, actions, next_states, rewards = [], [], [], []
        for t in transitions:
            states.append(t.state)
            actions.append(t.action)
            next_states.append(t.next_state)
            rewards.append(t.reward)
        self.fit(states, actions, next_states, rewards)

    # prediction ---------------------------------------------------------------
    def predict(
        self, state: Sequence[float], action: Sequence[float]
    ) -> tuple[np.ndarray, float]:
        """Predict next state and reward for ``state`` and ``action``."""
        if np is None or self._W is None or self._b is None:  # pragma: no cover
            raise RuntimeError("Model is untrained or numpy missing")
        X = self._features(np.array(state), np.array(action))
        Y = X @ np.vstack([self._W, self._b])
        next_state = Y[0, : self.state_dim]
        reward = float(Y[0, -1])
        return next_state, reward


class WorldModelEnv(gym.Env if gym else object):
    """Tiny environment driven purely by a :class:`WorldModel`."""

    def __init__(self, model: WorldModel, observation_space, action_space) -> None:
        self.model = model
        self.observation_space = observation_space
        self.action_space = action_space
        if np is None:  # pragma: no cover - numpy required for env state
            raise RuntimeError("numpy is required for WorldModelEnv")
        self.state = np.zeros(observation_space.shape, dtype=float)

    def reset(self):  # type: ignore[override]
        self.state = np.zeros_like(self.state)
        return self.state

    def step(self, action):  # type: ignore[override]
        next_state, reward = self.model.predict(self.state, action)
        self.state = next_state
        done = False
        return next_state, reward, done, {}


__all__ = ["WorldModel", "WorldModelEnv", "Transition"]
