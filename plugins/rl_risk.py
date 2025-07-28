from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Sequence

import gym
from gym import spaces
from stable_baselines3 import PPO

from . import register_risk_check


class RiskEnv(gym.Env):
    """Simple environment that learns a position size multiplier."""

    def __init__(self, returns: Sequence[float], lookback: int = 10, max_size: float = 1.0) -> None:
        super().__init__()
        self.returns = np.asarray(returns, dtype=np.float32)
        self.lookback = lookback
        self.max_size = max_size
        self.action_space = spaces.Box(low=0.0, high=max_size, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(lookback + 2,), dtype=np.float32)
        self.reset()

    def reset(self):  # type: ignore[override]
        self.i = self.lookback
        self.position = 0.0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        window = self.returns[self.i - self.lookback : self.i]
        vol = np.std(window) if len(window) > 0 else 0.0
        cum = np.cumprod(1 + window)
        dd = 0.0
        if len(cum) > 0:
            peak = np.maximum.accumulate(cum)
            dd = (cum[-1] - peak[-1]) / peak[-1]
        obs = np.concatenate([window, [vol, dd]]).astype(np.float32)
        return obs

    def step(self, action):  # type: ignore[override]
        action = float(np.clip(action, 0.0, self.max_size))
        ret = self.returns[self.i]
        reward = action * ret - abs(action) * 0.001
        self.position = action
        self.i += 1
        done = self.i >= len(self.returns)
        return self._get_obs(), reward, done, {}


_POLICY = None


def _load_policy() -> PPO | None:
    global _POLICY
    if _POLICY is not None:
        return _POLICY
    path = Path(__file__).resolve().parents[1] / "models" / "rl_risk_policy.zip"
    if path.exists():
        _POLICY = PPO.load(path)
    return _POLICY


@register_risk_check
def calculate_position_size(returns: Sequence[float]) -> float:
    """Return a position size multiplier predicted by the RL policy."""
    policy = _load_policy()
    if policy is None:
        return 1.0
    arr = np.asarray(returns, dtype=np.float32)
    if len(arr) < 1:
        return 1.0
    lookback = getattr(policy, "lookback", 10)
    window = arr[-lookback:]
    vol = np.std(window) if len(window) > 0 else 0.0
    cum = np.cumprod(1 + window)
    dd = 0.0
    if len(cum) > 0:
        peak = np.maximum.accumulate(cum)
        dd = (cum[-1] - peak[-1]) / peak[-1]
    obs = np.concatenate([window[-lookback:], [vol, dd]]).astype(np.float32)
    action, _ = policy.predict(obs, deterministic=True)
    size = float(np.clip(action, 0.0, 1.0))
    return size

