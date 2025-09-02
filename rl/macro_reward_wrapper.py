from __future__ import annotations

import numpy as np
import gym

from analytics.metrics_store import record_metric, TS_PATH


class MacroRewardWrapper(gym.Wrapper):
    """Augment rewards with macro feature alignment bonus.

    Provides a small reward when the agent's net position aligns with
    the average sign of the macro features at each step.
    """

    def __init__(self, env: gym.Env, bonus_coeff: float = 0.01) -> None:
        super().__init__(env)
        self.bonus_coeff = bonus_coeff

    def step(self, action):  # type: ignore[override]
        obs, reward, done, info = self.env.step(action)
        macro_bonus = 0.0
        macro_features = getattr(self.env, "macro_features", [])
        if macro_features:
            idx = getattr(self.env, "i", 0) - 1
            if idx >= 0:
                vals = self.env.df.loc[idx, macro_features].values.astype(np.float32)
                trend = float(np.mean(vals))
                exposure = float(np.sum(getattr(self.env, "positions", 0.0)))
                if trend != 0 and exposure != 0:
                    macro_bonus = self.bonus_coeff * np.sign(trend) * np.sign(exposure) * abs(exposure)
                    reward += macro_bonus
        info["macro_bonus"] = macro_bonus
        info["macro_adjusted_reward"] = reward
        record_metric("macro_bonus", float(macro_bonus), path=TS_PATH)
        record_metric("macro_adjusted_reward", float(reward), path=TS_PATH)
        return obs, reward, done, info


__all__ = ["MacroRewardWrapper"]
