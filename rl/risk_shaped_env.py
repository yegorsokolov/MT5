from __future__ import annotations

import numpy as np

from .trading_env import TradingEnv


class RiskShapedTradingEnv(TradingEnv):
    """Trading environment that subtracts risk related costs from reward.

    Additional penalties can be applied for drawdown, portfolio volatility and
    estimated slippage when positions change.
    """

    def __init__(
        self,
        *args,
        drawdown_penalty: float = 0.0,
        vol_penalty: float = 0.0,
        slippage_penalty: float = 0.0,
        vol_window: int = 30,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.drawdown_penalty = drawdown_penalty
        self.vol_penalty = vol_penalty
        self.slippage_penalty = slippage_penalty
        self.vol_window = vol_window

    def step(self, action):  # type: ignore[override]
        prev_positions = self.positions.copy()
        obs, reward, done, info = super().step(action)

        penalty = 0.0
        if self.drawdown_penalty > 0:
            drawdown = max(0.0, (self.peak_equity - self.equity) / self.peak_equity)
            dd_cost = self.drawdown_penalty * drawdown
            penalty += dd_cost
            info["drawdown_cost"] = dd_cost
        if self.vol_penalty > 0 and len(self.portfolio_returns) >= max(2, self.vol_window):
            window = np.asarray(self.portfolio_returns[-self.vol_window :])
            vol = float(np.std(window))
            vol_cost = self.vol_penalty * vol
            penalty += vol_cost
            info["volatility_cost"] = vol_cost
        if self.slippage_penalty > 0:
            deltas = self.positions - prev_positions
            slip_cost = float(np.abs(deltas).sum() * self.slippage_penalty)
            penalty += slip_cost
            info["slippage_cost"] = slip_cost

        reward -= penalty
        return obs, reward, done, info


__all__ = ["RiskShapedTradingEnv"]
