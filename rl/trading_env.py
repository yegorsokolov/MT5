from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import gym
from gym import spaces

try:  # optional torch dependency
    import torch
    from models.orderbook_gnn import OrderBookGNN, build_orderbook_graph
except Exception:  # pragma: no cover - torch may be missing
    torch = None  # type: ignore
    OrderBookGNN = build_orderbook_graph = None  # type: ignore

from analytics.metrics_store import record_metric, TS_PATH
from rl.multi_objective import weighted_sum


class TradingEnv(gym.Env):
    """Trading environment supporting multiple symbols with close action."""

    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        macro_features: List[str] | None = None,
        news_window: int = 0,
        orderbook_depth: int = 0,
        use_orderbook_gnn: bool | None = None,
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
        vector_feats = [
            c
            for c in df.columns
            if c.startswith("news_sentiment_") or c.startswith("news_impact_")
        ]

        base_cols = features + ["mid"] + vector_feats
        wide = df.set_index(["Timestamp", "Symbol"])[base_cols].unstack("Symbol")
        wide.columns = [f"{sym}_{feat}" for feat, sym in wide.columns]
        wide = wide.dropna()

        self.news_window = news_window
        self.news_feature_cols: list[str] = []
        if news_window > 0:
            try:
                news_wide = (
                    df.set_index(["Timestamp", "Symbol"])["news_movement_score"]
                    .unstack("Symbol")
                    .reindex(wide.index)
                    .fillna(0.0)
                )
            except KeyError:
                news_wide = pd.DataFrame(0.0, index=wide.index, columns=self.symbols)
            for sym in self.symbols:
                for k in range(news_window):
                    col = f"{sym}_news_{k}"
                    wide[col] = news_wide[sym].shift(k).fillna(0.0).values
                    self.news_feature_cols.append(col)

        # Include precomputed news sentiment/impact vectors if present
        for feat in vector_feats:
            for sym in self.symbols:
                col = f"{sym}_{feat}"
                if col in wide.columns:
                    self.news_feature_cols.append(col)

        macro_features = macro_features or []
        self.features = features
        self.macro_features = macro_features
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
        self.orderbook_depth = orderbook_depth
        self.use_orderbook_gnn = use_orderbook_gnn
        self.orderbook_cols: list[str] = []
        self.orderbook_gnn = None
        self.embedding_dim = 0
        self.train_orderbook_gnn = False
        self.device = None
        if orderbook_depth > 0:
            for sym in self.symbols:
                for side in ("bid", "ask"):
                    for lvl in range(orderbook_depth):
                        for field in ("px", "sz"):
                            col = f"{sym}_{side}_{field}_{lvl}"
                            if col not in self.df.columns:
                                raise ValueError(f"Missing column {col}")
                            self.orderbook_cols.append(col)
            if self.use_orderbook_gnn is None:
                self.use_orderbook_gnn = bool(torch and torch.cuda.is_available())
            if (
                self.use_orderbook_gnn
                and torch is not None
                and OrderBookGNN is not None
                and torch.cuda.is_available()
            ):
                self.device = torch.device("cuda")
                self.orderbook_gnn = OrderBookGNN(in_channels=3, hidden_channels=16).to(
                    self.device
                )
                self.embedding_dim = self.orderbook_gnn.hidden_channels
                self.train_orderbook_gnn = True
            else:
                self.use_orderbook_gnn = False
                self.embedding_dim = orderbook_depth * 4
                self.device = torch.device("cpu") if torch is not None else None
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
        self.feature_cols.extend(macro_features)
        self.feature_cols.extend(self.news_feature_cols)
        if self.orderbook_depth > 0 and not self.use_orderbook_gnn:
            self.feature_cols.extend(self.orderbook_cols)
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
        try:
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
        except AttributeError:  # pragma: no cover - gym stub without Dict
            self.action_space = spaces.Box(
                low=-max_position,
                high=max_position,
                shape=(self.n_symbols * 2,),
                dtype=np.float32,
            )
        obs_dim = len(self.feature_cols)
        if self.orderbook_depth > 0 and self.use_orderbook_gnn:
            obs_dim += self.n_symbols * self.embedding_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
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
        base = self.df.loc[self.i, self.feature_cols].values.astype(np.float32)
        if self.orderbook_depth > 0 and self.use_orderbook_gnn:
            base = np.concatenate([base, self._orderbook_embedding(self.i)])
        return base

    def _orderbook_embedding(self, idx: int) -> np.ndarray:
        embeds: list[np.ndarray] = []
        if torch is None or self.orderbook_gnn is None:
            return np.zeros(self.n_symbols * self.embedding_dim, dtype=np.float32)
        for sym in self.symbols:
            bids = []
            asks = []
            for lvl in range(self.orderbook_depth):
                bids.append(
                    [
                        float(self.df.loc[idx, f"{sym}_bid_px_{lvl}"]),
                        float(self.df.loc[idx, f"{sym}_bid_sz_{lvl}"]),
                    ]
                )
                asks.append(
                    [
                        float(self.df.loc[idx, f"{sym}_ask_px_{lvl}"]),
                        float(self.df.loc[idx, f"{sym}_ask_sz_{lvl}"]),
                    ]
                )
            bids_t = torch.tensor(bids, dtype=torch.float32, device=self.device)
            asks_t = torch.tensor(asks, dtype=torch.float32, device=self.device)
            x, edge_index = build_orderbook_graph(bids_t, asks_t)
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
            if self.train_orderbook_gnn:
                emb = self.orderbook_gnn(x, edge_index)
            else:
                with torch.no_grad():
                    emb = self.orderbook_gnn(x, edge_index)
            embeds.append(emb.detach().cpu().numpy())
        return np.concatenate(embeds).astype(np.float32)

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
        if self.orderbook_depth > 0 and self.use_orderbook_gnn:
            next_obs = np.concatenate([next_obs, self._orderbook_embedding(self.i)])

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
