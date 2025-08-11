"""Gym wrapper applying Conditional Value at Risk penalty to rewards."""

from __future__ import annotations

import numpy as np
import gym


class CVaRRewardWrapper(gym.Wrapper):
    """Augment environment rewards with a CVaR based penalty.

    Parameters
    ----------
    env:
        Environment to wrap.
    penalty: float, default 0.0
        Multiplicative factor for the CVaR penalty.
    window: int, default 30
        Number of recent returns to compute CVaR over.
    alpha: float, default 0.05
        Tail probability level used for the Value at Risk threshold.
    """

    def __init__(
        self,
        env: gym.Env,
        penalty: float = 0.0,
        window: int = 30,
        alpha: float = 0.05,
    ) -> None:
        super().__init__(env)
        self.penalty = penalty
        self.window = window
        self.alpha = alpha
        self._returns: list[float] = []

    def reset(self, **kwargs):  # type: ignore[override]
        self._returns.clear()
        return self.env.reset(**kwargs)

    def step(self, action):  # type: ignore[override]
        obs, reward, done, info = self.env.step(action)
        ret = info.get("portfolio_return", reward)
        self._returns.append(float(ret))

        cvar = 0.0
        if self.penalty > 0 and len(self._returns) >= self.window:
            window = np.asarray(self._returns[-self.window :])
            var = np.quantile(window, self.alpha)
            tail = window[window <= var]
            if tail.size:
                cvar = -float(tail.mean())
                reward -= self.penalty * cvar
        info["cvar"] = cvar
        return obs, reward, done, info


__all__ = ["CVaRRewardWrapper"]
