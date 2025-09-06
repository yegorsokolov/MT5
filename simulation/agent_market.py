"""Agent-based market simulator for offline experimentation.

This module implements a very small synthetic market consisting of three
participant types:

* ``MarketMaker`` posts bid/ask quotes around the current mid price.
* ``TrendFollower`` trades in the direction of recent price movements.
* ``LiquidityTaker`` submits random buy/sell market orders.

The interaction of these agents generates a price path and a simple order
book.  The resulting trade and price series are persisted under
``data/simulations`` to enable reproducible offline training and validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class AgentMarketSimulator:
    """Simple agent-based market simulator.

    Parameters
    ----------
    seed: int
        Random seed for reproducibility.
    steps: int
        Number of simulation steps to generate.
    start_price: float, optional
        Initial mid price.
    out_dir: str | Path, optional
        Directory where ``prices`` and ``trades`` CSV files are saved.
    """

    seed: int = 42
    steps: int = 1000
    start_price: float = 100.0
    out_dir: str | Path = "data/simulations"

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the market simulation.

        Returns
        -------
        trades: pd.DataFrame
            Executed trades with ``step``, ``price``, ``volume`` and ``side``.
        book: pd.DataFrame
            Order book with ``step``, ``bid``, ``ask`` and ``mid`` prices.
        """

        rng = np.random.default_rng(self.seed)
        price = float(self.start_price)
        trades: list[dict] = []
        book: list[dict] = []
        trend = 0.0
        for step in range(self.steps):
            spread = 0.1
            bid = price - spread / 2
            ask = price + spread / 2
            # Trend follower acts on previous price movement
            trend_order = np.sign(trend)
            # Liquidity taker submits random order
            lt_order = rng.integers(-1, 2)
            net_order = trend_order + lt_order
            noise = rng.normal(scale=0.01)
            prev_price = price
            price = price + 0.01 * net_order + noise
            side = "buy" if net_order > 0 else "sell" if net_order < 0 else "flat"
            trade_price = ask if net_order > 0 else bid if net_order < 0 else price
            volume = abs(net_order)
            trades.append(
                {
                    "step": step,
                    "price": trade_price,
                    "volume": volume,
                    "side": side,
                }
            )
            book.append({"step": step, "bid": bid, "ask": ask, "mid": price})
            trend = price - prev_price

        trades_df = pd.DataFrame(trades)
        book_df = pd.DataFrame(book)

        out_path = Path(self.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        trades_path = out_path / f"trades_seed{self.seed}.csv"
        prices_path = out_path / f"prices_seed{self.seed}.csv"
        trades_df.to_csv(trades_path, index=False)
        book_df.to_csv(prices_path, index=False)
        return trades_df, book_df

    def to_history_df(self, book: pd.DataFrame) -> pd.DataFrame:
        """Convert book data to OHLCV format for training environments."""

        ts = pd.date_range("2020-01-01", periods=len(book), freq="T")
        df = pd.DataFrame({
            "Timestamp": ts,
            "Open": book["mid"].values,
            "High": book["mid"].values + 0.01,
            "Low": book["mid"].values - 0.01,
            "Close": book["mid"].values,
            "Volume": 1.0,
        })
        return df
