from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import gym
from gym import spaces

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    class MultiAgentEnv(gym.Env):
        """Fallback stub when RLlib is not available."""

        pass

from rl.multi_objective import weighted_sum


class MultiAgentTradingEnv(MultiAgentEnv):
    """Multi-agent trading environment with shared rewards.

    Each agent controls a single instrument. Agents receive rewards composed of
    their individual PnL plus shared components such as portfolio return and
    risk penalty.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        macro_features: List[str] | None = None,
        max_position: float = 1.0,
        transaction_cost: float = 0.0001,
        risk_penalty: float = 0.1,
        var_window: int = 30,
    ) -> None:
        super().__init__()
        if "Symbol" not in df.columns:
            raise ValueError("DataFrame must contain a 'Symbol' column")
        self.symbols = sorted(df["Symbol"].unique())
        wide = df.set_index(["Timestamp", "Symbol"])[features + ["mid"]].unstack("Symbol")
        wide.columns = [f"{sym}_{feat}" for feat, sym in wide.columns]
        wide = wide.dropna()

        macro_features = macro_features or []
        macro_df = pd.DataFrame()
        if macro_features:
            macro_df = (
                df.set_index("Timestamp")[macro_features]
                .groupby("Timestamp")
                .first()
            )
            if not macro_df.empty:
                wide = wide.join(macro_df, how="left")

        self.df = wide.reset_index(drop=True)
        self.features = features
        self.macro_features = macro_features
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.risk_penalty = risk_penalty
        self.var_window = var_window

        self.price_cols = [f"{sym}_mid" for sym in self.symbols]
        self.feature_cols: Dict[str, List[str]] = {
            sym: [f"{sym}_{feat}" for feat in features] + macro_features
            for sym in self.symbols
        }
        self.n_symbols = len(self.symbols)
        self.agents = list(self.symbols)

        self.action_space = spaces.Dict(
            {
                sym: spaces.Box(
                    low=-max_position,
                    high=max_position,
                    shape=(1,),
                    dtype=np.float32,
                )
                for sym in self.symbols
            }
        )
        self.observation_space = spaces.Dict(
            {
                sym: spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(features) + len(macro_features),),
                    dtype=np.float32,
                )
                for sym in self.symbols
            }
        )
        self.start_equity = 1.0
        self.reset()

    # RLlib MultiAgentEnv style
    def reset(self, *, seed: int | None = None, options=None):  # type: ignore[override]
        self.i = 0
        self.equity = self.start_equity
        self.peak_equity = self.start_equity
        self.positions = np.zeros(self.n_symbols, dtype=np.float32)
        self.portfolio_returns: List[float] = []
        obs = {
            sym: self.df.loc[self.i, cols].values.astype(np.float32)
            for sym, cols in self.feature_cols.items()
        }
        return obs

    def step(self, actions: Dict[str, float]):  # type: ignore[override]
        done = False
        size = np.zeros(self.n_symbols, dtype=np.float32)
        for idx, sym in enumerate(self.symbols):
            act = actions.get(sym, 0.0)
            size[idx] = np.clip(float(act), -self.max_position, self.max_position)
        prices = self.df.loc[self.i, self.price_cols].values

        portfolio_ret = 0.0
        per_symbol_ret = np.zeros(self.n_symbols, dtype=np.float32)
        if self.i > 0:
            prev_prices = self.df.loc[self.i - 1, self.price_cols].values
            price_change = (prices - prev_prices) / prev_prices
            per_symbol_ret = self.positions * price_change
            portfolio_ret = per_symbol_ret.sum()
            self.equity *= 1 + portfolio_ret

        deltas = size - self.positions
        costs = np.abs(deltas) * self.transaction_cost
        self.equity *= 1 - costs.sum()

        self.positions = size
        self.i += 1
        if self.i >= len(self.df) - 1:
            done = True
        next_obs = {
            sym: self.df.loc[self.i, cols].values.astype(np.float32)
            for sym, cols in self.feature_cols.items()
        }

        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = (self.equity - self.peak_equity) / self.peak_equity
        self.portfolio_returns.append(portfolio_ret)
        risk = -abs(drawdown)
        if len(self.portfolio_returns) >= self.var_window:
            var = np.var(self.portfolio_returns[-self.var_window :])
            risk -= self.risk_penalty * var
        shared = portfolio_ret + risk

        rewards = {
            sym: float(per_symbol_ret[idx] + shared)
            for idx, sym in enumerate(self.symbols)
        }
        dones = {sym: done for sym in self.symbols}
        dones["__all__"] = done
        infos = {
            sym: {"portfolio_return": float(portfolio_ret), "risk": float(risk)}
            for sym in self.symbols
        }
        return next_obs, rewards, dones, infos


__all__ = ["MultiAgentTradingEnv"]
