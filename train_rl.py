from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import gym
from gym import spaces
from stable_baselines3 import PPO

from utils import load_config
from dataset import load_history, load_history_from_urls, make_features


class TradingEnv(gym.Env):
    """Trading environment with long/short positions and transaction costs."""

    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        max_position: float = 1.0,
        transaction_cost: float = 0.0001,
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.features = features
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.action_space = spaces.Box(
            low=-max_position, high=max_position, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(features),), dtype=np.float32
        )
        self.start_equity = 1.0
        self.reset()

    def reset(self):
        self.i = 0
        self.equity = self.start_equity
        self.peak_equity = self.start_equity
        self.position = 0.0
        obs = self.df.loc[self.i, self.features].values.astype(np.float32)
        return obs

    def step(self, action):
        done = False
        reward = 0.0

        action = float(np.clip(action, -self.max_position, self.max_position))
        price = self.df.loc[self.i, "mid"]

        if self.i > 0:
            prev_price = self.df.loc[self.i - 1, "mid"]
            price_change = (price - prev_price) / prev_price
            reward += self.position * price_change
            self.equity *= 1 + self.position * price_change

        cost = abs(action - self.position) * self.transaction_cost
        reward -= cost
        self.equity *= 1 - cost
        self.position = action

        self.i += 1
        if self.i >= len(self.df) - 1:
            done = True
        next_obs = self.df.loc[self.i, self.features].values.astype(np.float32)

        # risk penalty based on drawdown
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = (self.equity - self.peak_equity) / self.peak_equity
        reward += -abs(drawdown) * 0.1

        return next_obs, reward, done, {}


def main():
    cfg = load_config()
    root = Path(__file__).resolve().parent
    root.joinpath("data").mkdir(exist_ok=True)

    symbols = cfg.get("symbols") or [cfg.get("symbol")]
    dfs = []
    for sym in symbols:
        path = root / "data" / f"{sym}_history.csv"
        if path.exists():
            df_sym = load_history(path)
        else:
            urls = cfg.get("data_urls", {}).get(sym)
            if not urls:
                raise FileNotFoundError(f"No history found for {sym} and no URL configured")
            df_sym = load_history_from_urls(urls)
            df_sym.to_csv(path, index=False)
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
    ]
    if "volume_ratio" in df.columns:
        features.extend(["volume_ratio", "volume_imbalance"])

    env = TradingEnv(df, features)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=cfg.get("rl_steps", 5000))
    model.save(root / "model_rl")
    print("RL model saved to", root / "model_rl.zip")


if __name__ == "__main__":
    main()
