"""Reinforcement learning based execution simulator.

This module provides a light-weight environment that simulates a
limit-order book (LOB) using historical snapshots. The environment can
be used to train a policy that decides when to slice a large parent
order and at which price level to submit the slice. It intentionally
avoids heavyweight dependencies so unit tests can exercise the basic
logic without requiring specialised hardware.

The implementation is deliberately simple – it only models the top of
book and a single decision per step (execute or wait). Policies trained
on the environment can nevertheless learn basic execution tactics such
as waiting for favourable liquidity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from rl.gym_compat import gym, spaces

try:  # optional dependency used for feature engineering
    from data.order_book import compute_order_book_features, load_order_book
except Exception:  # pragma: no cover - data package may be stubbed in tests
    def load_order_book(source):  # type: ignore
        return pd.DataFrame(source)

    def compute_order_book_features(df: pd.DataFrame) -> pd.DataFrame:  # type: ignore
        return df


class LOBExecutionEnv(gym.Env if gym else object):
    """A tiny order execution environment based on LOB snapshots."""

    metadata = {"render.modes": []}

    def __init__(self, book: pd.DataFrame, side: str = "buy", slice_size: float = 1.0):
        if not gym:  # pragma: no cover - executed only when gym missing
            raise RuntimeError("gym is required for LOBExecutionEnv")
        self.book = book.reset_index(drop=True)
        self.side = side.lower()
        self.slice_size = slice_size
        self.current_step = 0
        self.remaining = 1.0
        self.action_space = spaces.Discrete(2)  # 0 = wait, 1 = execute
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        row = self.book.iloc[self.current_step]
        return np.array(
            [
                row["depth_imbalance"],
                row["vw_spread"],
                row["liquidity"],
                row["slippage"],
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.current_step = 0
        self.remaining = 1.0
        return self._get_obs()

    # ------------------------------------------------------------------
    def step(self, action: int):
        row = self.book.iloc[self.current_step]
        mid = (row["BidPrice1"] + row["AskPrice1"]) / 2.0
        reward = 0.0
        done = False
        if action == 1 and self.remaining > 0:
            price = row["AskPrice1"] if self.side == "buy" else row["BidPrice1"]
            # Reward is negative slippage from mid price
            reward = -(price - mid) if self.side == "buy" else -(mid - price)
            self.remaining -= self.slice_size
        self.current_step += 1
        if self.current_step >= len(self.book) or self.remaining <= 0:
            done = True
        return self._get_obs(), float(reward), done, {}


@dataclass
class RLExecutor:
    """Helper that wraps a learnt policy for order execution."""

    env: Optional[LOBExecutionEnv] = None
    model: Optional[object] = None

    # ------------------------------------------------------------------
    @staticmethod
    def make_env(source: str | Path | pd.DataFrame, side: str = "buy") -> LOBExecutionEnv:
        """Create :class:`LOBExecutionEnv` from an order book source."""
        df = load_order_book(source)
        df = compute_order_book_features(df)
        return LOBExecutionEnv(df, side=side)

    # ------------------------------------------------------------------
    def train(self, steps: int = 1000) -> None:
        """Train a policy on the attached environment.

        The routine attempts to use ``sb3_contrib.QRDQN`` when available to
        provide a simple Deep‑Q implementation. If the dependency is
        missing, a very small tabular Q‑learning fallback is used instead
        which is sufficient for tests.
        """
        if self.env is None:
            raise ValueError("Environment required for training")
        try:  # preferred path using stable baselines
            from sb3_contrib.qrdqn import QRDQN  # type: ignore

            self.model = QRDQN("MlpPolicy", self.env, verbose=0)
            self.model.learn(total_timesteps=int(steps))
            return
        except Exception:  # pragma: no cover - fallback path
            pass

        # ------------------------------------------------------------------
        # Basic tabular Q-learning fallback
        n_states = len(self.env.book)
        n_actions = int(self.env.action_space.n)
        q = np.zeros((n_states + 1, n_actions))
        gamma = 0.95
        alpha = 0.1
        eps = 0.1
        for _ in range(int(steps)):
            _ = self.env.reset()
            s = self.env.current_step
            done = False
            while not done:
                if np.random.rand() < eps:
                    a = self.env.action_space.sample()
                else:
                    a = int(np.argmax(q[s]))
                _, r, done, _ = self.env.step(a)
                s_next = self.env.current_step
                q[s, a] += alpha * (r + gamma * np.max(q[s_next]) - q[s, a])
                s = s_next
        self.model = q  # store q-table

    # ------------------------------------------------------------------
    def _policy_action(self, obs: np.ndarray) -> int:
        if self.model is None:
            return 1  # default: execute immediately
        if hasattr(self.model, "predict"):
            act, _ = self.model.predict(obs, deterministic=True)
            return int(act[0] if np.ndim(act) else act)
        # treat as tabular q-table
        idx = int(getattr(self.env, "current_step", 0))
        return int(np.argmax(self.model[idx]))

    # ------------------------------------------------------------------
    def _obs_from_snapshot(
        self, bid: float, ask: float, bid_vol: float, ask_vol: float
    ) -> np.ndarray:
        df = pd.DataFrame(
            {
                "Timestamp": [0],
                "BidPrice1": [bid],
                "AskPrice1": [ask],
                "BidVolume1": [bid_vol],
                "AskVolume1": [ask_vol],
            }
        )
        feats = compute_order_book_features(df)
        row = feats.iloc[0]
        return np.array(
            [row["depth_imbalance"], row["vw_spread"], row["liquidity"], row["slippage"]],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    def execute(
        self,
        *,
        side: str,
        quantity: float,
        bid: float,
        ask: float,
        bid_vol: float,
        ask_vol: float,
        mid: float,
    ) -> dict:
        """Execute ``quantity`` according to the trained policy."""
        obs = self._obs_from_snapshot(bid, ask, bid_vol, ask_vol)
        act = self._policy_action(obs)
        if act == 0:  # wait
            return {"avg_price": mid, "filled": 0.0}
        avail = ask_vol if side.lower() == "buy" else bid_vol
        price = ask if side.lower() == "buy" else bid
        fill_qty = min(quantity, avail)
        return {"avg_price": price, "filled": fill_qty}

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        if self.model is None:
            return
        try:
            if hasattr(self.model, "save"):
                self.model.save(str(path))
            else:  # q-table
                np.save(path, self.model)
        except Exception:  # pragma: no cover - best effort
            pass

    # ------------------------------------------------------------------
    def load(self, path: str | Path, env: Optional[LOBExecutionEnv] = None) -> None:
        if env is not None:
            self.env = env
        try:
            from sb3_contrib.qrdqn import QRDQN  # type: ignore

            self.model = QRDQN.load(str(path), env=self.env)
            return
        except Exception:  # pragma: no cover - fallback
            try:
                self.model = np.load(path, allow_pickle=True)
            except Exception:
                self.model = None


__all__ = ["RLExecutor", "LOBExecutionEnv"]
