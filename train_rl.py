from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import gym
from gym import spaces
from stable_baselines3 import PPO
from sb3_contrib.qrdqn import QRDQN

from utils import load_config
from dataset import (
    load_history,
    load_history_from_urls,
    load_history_parquet,
    save_history_parquet,
    make_features,
)

from log_utils import setup_logging, log_exceptions
logger = setup_logging()


class TradingEnv(gym.Env):
    """Trading environment supporting multiple symbols."""

    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        max_position: float = 1.0,
        transaction_cost: float = 0.0001,
        risk_penalty: float = 0.1,
        var_window: int = 30,
    ) -> None:
        super().__init__()

        if "Symbol" not in df.columns:
            raise ValueError("DataFrame must contain a 'Symbol' column")
        self.symbols = sorted(df["Symbol"].unique())
        wide = (
            df.set_index(["Timestamp", "Symbol"])[features + ["mid"]]
            .unstack("Symbol")
        )
        wide.columns = [f"{sym}_{feat}" for feat, sym in wide.columns]
        wide = wide.dropna().reset_index(drop=True)

        self.df = wide
        self.features = features
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.risk_penalty = risk_penalty
        self.var_window = var_window

        self.price_cols = [f"{sym}_mid" for sym in self.symbols]
        self.feature_cols = []
        for feat in features:
            self.feature_cols.extend([f"{sym}_{feat}" for sym in self.symbols])

        self.n_symbols = len(self.symbols)
        self.action_space = spaces.Box(
            low=-max_position,
            high=max_position,
            shape=(self.n_symbols,),
            dtype=np.float32,
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

        action = np.clip(np.asarray(action, dtype=np.float32), -self.max_position, self.max_position)
        prices = self.df.loc[self.i, self.price_cols].values

        portfolio_ret = 0.0
        per_symbol_ret = np.zeros(self.n_symbols, dtype=np.float32)
        if self.i > 0:
            prev_prices = self.df.loc[self.i - 1, self.price_cols].values
            price_change = (prices - prev_prices) / prev_prices
            per_symbol_ret = self.positions * price_change
            portfolio_ret = per_symbol_ret.sum()
            self.equity *= 1 + portfolio_ret

        costs = np.abs(action - self.positions) * self.transaction_cost
        cost_total = costs.sum()
        self.equity *= 1 - cost_total

        reward = portfolio_ret - cost_total
        self.positions = action

        self.i += 1
        if self.i >= len(self.df) - 1:
            done = True
        next_obs = self.df.loc[self.i, self.feature_cols].values.astype(np.float32)

        # risk penalty based on drawdown and variance
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = (self.equity - self.peak_equity) / self.peak_equity

        self.portfolio_returns.append(portfolio_ret)
        risk = -abs(drawdown) * 0.1
        if len(self.portfolio_returns) >= self.var_window:
            var = np.var(self.portfolio_returns[-self.var_window:])
            risk -= self.risk_penalty * var

        reward += risk

        info = {
            "portfolio_return": float(portfolio_ret),
            "per_symbol_returns": per_symbol_ret,
            "transaction_costs": costs,
        }

        return next_obs, reward, done, info


class DiscreteTradingEnv(TradingEnv):
    """Discrete version for QRDQN."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import itertools

        levels = [-self.max_position, 0.0, self.max_position]
        self.discrete_actions = np.array(
            list(itertools.product(levels, repeat=self.n_symbols)), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.discrete_actions))

    def step(self, action):
        continuous = self.discrete_actions[int(action)]
        return super().step(continuous)


@log_exceptions
def main():
    cfg = load_config()
    root = Path(__file__).resolve().parent
    root.joinpath("data").mkdir(exist_ok=True)

    symbols = cfg.get("symbols") or [cfg.get("symbol")]
    dfs = []
    for sym in symbols:
        csv_path = root / "data" / f"{sym}_history.csv"
        pq_path = root / "data" / f"{sym}_history.parquet"
        if pq_path.exists():
            df_sym = load_history_parquet(pq_path)
        elif csv_path.exists():
            df_sym = load_history(csv_path)
        else:
            urls = cfg.get("data_urls", {}).get(sym)
            if not urls:
                raise FileNotFoundError(f"No history found for {sym} and no URL configured")
            df_sym = load_history_from_urls(urls)
            save_history_parquet(df_sym, pq_path)
        df_sym["Symbol"] = sym
        dfs.append(df_sym)

    df = make_features(pd.concat(dfs, ignore_index=True))
    features = [
        "return",
        "ma_5",
        "ma_10",
        "ma_30",
        "ma_60",
        "volatility_30",
        "spread",
        "rsi_14",
        "news_sentiment",
        "market_regime",
    ]
    features += [
        c
        for c in df.columns
        if c.startswith("cross_corr_")
        or c.startswith("factor_")
        or c.startswith("cross_mom_")
    ]
    if "volume_ratio" in df.columns:
        features.extend(["volume_ratio", "volume_imbalance"])

    algo = cfg.get("rl_algorithm", "PPO").upper()
    if algo == "PPO":
        env = TradingEnv(
            df,
            features,
            max_position=cfg.get("rl_max_position", 1.0),
            transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
            risk_penalty=cfg.get("rl_risk_penalty", 0.1),
            var_window=cfg.get("rl_var_window", 30),
        )
        model = PPO("MlpPolicy", env, verbose=0)
    elif algo == "QRDQN":
        env = DiscreteTradingEnv(
            df,
            features,
            max_position=cfg.get("rl_max_position", 1.0),
            transaction_cost=cfg.get("rl_transaction_cost", 0.0001),
            risk_penalty=cfg.get("rl_risk_penalty", 0.1),
            var_window=cfg.get("rl_var_window", 30),
        )
        model = QRDQN("MlpPolicy", env, verbose=0)
    else:
        raise ValueError(f"Unknown rl_algorithm {algo}")

    model.learn(total_timesteps=cfg.get("rl_steps", 5000))
    model.save(root / "model_rl")
    print("RL model saved to", root / "model_rl.zip")


if __name__ == "__main__":
    main()
