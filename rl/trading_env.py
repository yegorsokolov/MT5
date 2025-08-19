from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import gym
from gym import spaces

from analytics.metrics_store import record_metric, TS_PATH
from rl.multi_objective import weighted_sum


class TradingEnv(gym.Env):
    """Trading environment supporting multiple symbols with close action."""

    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        max_position: float = 1.0,
        transaction_cost: float = 0.0001,
        risk_penalty: float = 0.1,
        var_window: int = 30,
        cvar_penalty: float = 0.0,
        cvar_window: int = 30,
        slippage_factor: float = 0.0,
        spread_source: str | None = None,
        objectives: list[str] | None = None,
        objective_weights: list[float] | None = None,
        exit_penalty: float = 0.001,
    ) -> None:
        super().__init__()

        if "Symbol" not in df.columns:
            raise ValueError("DataFrame must contain a 'Symbol' column")
        self.symbols = sorted(df["Symbol"].unique())
        wide = df.set_index(["Timestamp", "Symbol"])[features + ["mid"]].unstack("Symbol")
        wide.columns = [f"{sym}_{feat}" for feat, sym in wide.columns]
        wide = wide.dropna().reset_index(drop=True)

        self.df = wide
        self.features = features
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.risk_penalty = risk_penalty
        self.var_window = var_window
        self.cvar_penalty = cvar_penalty
        self.cvar_window = cvar_window
        self.slippage_factor = slippage_factor
        self.spread_source = spread_source
        self.objectives = objectives or ["pnl", "hold_cost"]
        self.exit_penalty = exit_penalty
        if objective_weights is None:
            objective_weights = [1.0] * len(self.objectives)
        self.objective_weights = np.asarray(objective_weights, dtype=np.float32)

        self.price_cols = [f"{sym}_mid" for sym in self.symbols]
        self.feature_cols = []
        for feat in features:
            self.feature_cols.extend([f"{sym}_{feat}" for sym in self.symbols])
        self.spread_cols = []
        self.market_impact_cols: list[str | None] = []
        for sym in self.symbols:
            if f"{sym}_vw_spread" in self.df.columns:
                self.spread_cols.append(f"{sym}_vw_spread")
            else:
                self.spread_cols.append(f"{sym}_spread")
            self.market_impact_cols.append(
                f"{sym}_market_impact" if f"{sym}_market_impact" in self.df.columns else None
            )

        self.n_symbols = len(self.symbols)
        try:
            close_space = spaces.MultiBinary(self.n_symbols)
        except AttributeError:  # pragma: no cover - gym stub without MultiBinary
            close_space = spaces.Box(
                low=0, high=1, shape=(self.n_symbols,), dtype=np.int32
            )
        self.action_space = spaces.Dict(
            {
                "size": spaces.Box(
                    low=-max_position,
                    high=max_position,
                    shape=(self.n_symbols,),
                    dtype=np.float32,
                ),
                "close": close_space,
            }
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_cols),),
            dtype=np.float32,
        )
        self.start_equity = 1.0
        self.reset()

    def reset(self):
        self.i = 0
        self.equity = self.start_equity
        self.peak_equity = self.start_equity
        self.positions = np.zeros(self.n_symbols, dtype=np.float32)
        self.portfolio_returns: list[float] = []
        obs = self.df.loc[self.i, self.feature_cols].values.astype(np.float32)
        return obs

    def step(self, action):
        done = False

        if isinstance(action, dict):
            size = np.asarray(action.get("size", np.zeros(self.n_symbols)), dtype=np.float32)
            close = np.asarray(action.get("close", np.zeros(self.n_symbols)), dtype=np.float32)
        else:
            arr = np.asarray(action, dtype=np.float32).flatten()
            if arr.size == self.n_symbols:
                size = arr
                close = np.zeros(self.n_symbols, dtype=np.float32)
            elif arr.size == self.n_symbols * 2:
                size = arr[: self.n_symbols]
                close = arr[self.n_symbols :]
            else:
                raise ValueError("Invalid action shape")
        size = np.clip(size, -self.max_position, self.max_position)
        close = (close > 0.5).astype(np.float32)
        size = np.where(close == 1.0, 0.0, size)
        prices = self.df.loc[self.i, self.price_cols].values

        portfolio_ret = 0.0
        per_symbol_ret = np.zeros(self.n_symbols, dtype=np.float32)
        realized_pnl = 0.0
        if self.i > 0:
            prev_prices = self.df.loc[self.i - 1, self.price_cols].values
            price_change = (prices - prev_prices) / prev_prices
            per_symbol_ret = self.positions * price_change
            realized_pnl = float(np.sum(per_symbol_ret * close))
            portfolio_ret = per_symbol_ret.sum()
            self.equity *= 1 + portfolio_ret

        deltas = size - self.positions

        exec_prices = prices.copy()
        if self.spread_source == "column":
            try:
                spreads = self.df.loc[self.i, self.spread_cols].values
            except KeyError:
                spreads = np.zeros(self.n_symbols, dtype=np.float32)
            market_impacts = np.array(
                [self.df.loc[self.i, col] if col is not None else 0.0 for col in self.market_impact_cols],
                dtype=np.float32,
            )
            bids = prices - spreads / 2 - market_impacts
            asks = prices + spreads / 2 + market_impacts
            exec_prices = np.where(deltas > 0, asks, bids)

        slippage = np.zeros(self.n_symbols, dtype=np.float32)
        if self.slippage_factor > 0:
            slippage = np.abs(np.random.normal(scale=self.slippage_factor, size=self.n_symbols))
            exec_prices = np.where(
                deltas > 0,
                exec_prices * (1 + slippage),
                exec_prices * (1 - slippage),
            )

        transaction_costs = np.abs(deltas) * self.transaction_cost
        slippage_costs = np.abs(deltas) * np.abs(exec_prices - prices) / prices
        costs = transaction_costs + slippage_costs
        cost_total = costs.sum()
        self.equity *= 1 - cost_total
        record_metric("slippage", float(slippage_costs.sum()), path=TS_PATH)
        record_metric("liquidity_usage", float(np.abs(deltas).sum()), path=TS_PATH)

        opportunity_cost = float(np.sum(np.abs(per_symbol_ret * (1 - close))))
        reward_components: list[float] = []
        objective_map: dict[str, float] = {}
        if "pnl" in self.objectives:
            reward_components.append(realized_pnl)
            objective_map["pnl"] = realized_pnl
        if "hold_cost" in self.objectives:
            reward_components.append(-opportunity_cost)
            objective_map["hold_cost"] = -opportunity_cost
        if np.any(close):
            penalty = -self.exit_penalty * float(close.sum())
            reward_components.append(penalty)
            objective_map["exit"] = penalty
        if "cost" in self.objectives:
            reward_components.append(-cost_total)
            objective_map["cost"] = float(-cost_total)
        reward = 0.0
        self.positions = size

        self.i += 1
        if self.i >= len(self.df) - 1:
            done = True
        next_obs = self.df.loc[self.i, self.feature_cols].values.astype(np.float32)

        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = (self.equity - self.peak_equity) / self.peak_equity
        self.portfolio_returns.append(portfolio_ret)
        risk = -abs(drawdown) * 0.1
        if len(self.portfolio_returns) >= self.var_window:
            var = np.var(self.portfolio_returns[-self.var_window :])
            risk -= self.risk_penalty * var
        if len(self.portfolio_returns) >= self.cvar_window and self.cvar_penalty > 0:
            window = np.array(self.portfolio_returns[-self.cvar_window :])
            var_threshold = np.percentile(window, 5)
            cvar = -window[window <= var_threshold].mean()
            risk -= self.cvar_penalty * cvar
        if "risk" in self.objectives:
            reward_components.append(risk)
            objective_map["risk"] = float(risk)

        if reward_components:
            weights = np.ones(len(reward_components), dtype=np.float32)
            weights[: len(self.objective_weights)] = self.objective_weights[: len(self.objective_weights)]
            reward = weighted_sum(np.asarray(reward_components, dtype=np.float32), weights)

        info = {
            "portfolio_return": float(portfolio_ret),
            "per_symbol_returns": per_symbol_ret,
            "transaction_costs": costs,
            "execution_prices": exec_prices,
            "realized_pnl": realized_pnl,
            "opportunity_cost": opportunity_cost,
        }
        if objective_map:
            info["objectives"] = objective_map
        return next_obs, reward, done, info


class DiscreteTradingEnv(TradingEnv):
    """Discrete version for QRDQN including close action."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import itertools

        size_levels = [-self.max_position, 0.0, self.max_position]
        close_levels = [0.0, 1.0]
        actions = []
        for size_combo in itertools.product(size_levels, repeat=self.n_symbols):
            for close_combo in itertools.product(close_levels, repeat=self.n_symbols):
                actions.append(list(size_combo) + list(close_combo))
        self.discrete_actions = np.array(actions, dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.discrete_actions))

    def step(self, action):
        continuous = self.discrete_actions[int(action)]
        return super().step(continuous)


class HierarchicalTradingEnv(TradingEnv):
    """Environment with manager-worker actions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Dict(
            {
                "manager": spaces.Discrete(3),
                "worker": spaces.Box(
                    low=0.0,
                    high=self.max_position,
                    shape=(self.n_symbols,),
                    dtype=np.float32,
                ),
            }
        )

    def step(self, action):
        if isinstance(action, dict):
            manager = int(action.get("manager", 1))
            worker = np.asarray(action.get("worker", np.zeros(self.n_symbols)), dtype=np.float32)
        else:
            manager, worker = action
            worker = np.asarray(worker, dtype=np.float32)
        direction = {0: -1.0, 1: 0.0, 2: 1.0}.get(manager, 0.0)
        continuous = direction * np.clip(worker, 0.0, self.max_position)
        return super().step(continuous)


class RLLibTradingEnv(TradingEnv):
    """Wrapper returning gymnasium-style tuples for RLlib."""

    def reset(self, *, seed=None, options=None):  # type: ignore[override]
        obs = super().reset()
        return obs, {}

    def step(self, action):  # type: ignore[override]
        obs, reward, done, info = super().step(action)
        return obs, reward, done, False, info


__all__ = [
    "TradingEnv",
    "DiscreteTradingEnv",
    "HierarchicalTradingEnv",
    "RLLibTradingEnv",
]
