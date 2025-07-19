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
    """Simple trading environment for RL."""

    def __init__(self, df: pd.DataFrame, features: List[str]):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.features = features
        self.action_space = spaces.Discrete(2)  # 0 = flat, 1 = long
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(features),), dtype=np.float32
        )
        self.start_equity = 1.0
        self.reset()

    def reset(self):
        self.i = 0
        self.equity = self.start_equity
        self.peak_equity = self.start_equity
        self.position = 0
        self.entry_price = 0.0
        obs = self.df.loc[self.i, self.features].values.astype(np.float32)
        return obs

    def step(self, action):
        done = False
        reward = 0.0

        price = self.df.loc[self.i, "mid"]
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price
        elif action == 0 and self.position == 1:
            profit = (price - self.entry_price) / self.entry_price
            self.equity *= 1 + profit
            reward += profit
            self.position = 0

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
