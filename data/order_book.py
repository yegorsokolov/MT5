"""Utilities for working with Level II order book data.

This module provides helpers to load order book snapshots and compute
liquidity related features such as depth imbalance, volume weighted spread
and a simple market impact estimate.  It also derives aggregate *slippage*
and available *liquidity* measures which can be fed into risk controls.  The
functions are intentionally light‑weight and rely only on ``pandas`` making
them easy to unit test.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def load_order_book(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load historical Level II order book data.

    Parameters
    ----------
    source:
        Either a path to a CSV/Parquet file or a pre‑constructed dataframe.
        The dataframe is expected to contain a ``Timestamp`` column and level
        specific columns such as ``BidPrice1``, ``BidVolume1``, ``AskPrice1``
        and ``AskVolume1``.  Additional levels follow the same numbering
        pattern (``BidPrice2`` … ``BidPriceN``).
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        path = Path(source)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def _level_columns(prefix: str, columns: Iterable[str]) -> Sequence[str]:
    """Return all columns starting with ``prefix`` ordered by level number."""
    level_cols = [c for c in columns if c.startswith(prefix)]
    level_cols.sort(key=lambda x: int("".join(filter(str.isdigit, x)) or 0))
    return level_cols


def compute_order_book_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute liquidity features from order book snapshots.

    The input ``df`` must contain price and volume columns for both bid and ask
    sides.  Columns are detected automatically based on their prefixes
    (``BidPrice``, ``BidVolume``, ``AskPrice`` and ``AskVolume``).

    Returns
    -------
    pd.DataFrame
        Dataframe with ``depth_imbalance``, ``vw_spread``, ``market_impact``,
        ``slippage`` and ``liquidity`` columns appended.  ``slippage`` is a
        naive estimate of execution cost derived from the spread and the
        order book imbalance while ``liquidity`` represents the total depth
        available across bid and ask levels.
    """
    bid_price_cols = _level_columns("BidPrice", df.columns)
    bid_vol_cols = _level_columns("BidVolume", df.columns)
    ask_price_cols = _level_columns("AskPrice", df.columns)
    ask_vol_cols = _level_columns("AskVolume", df.columns)

    if not bid_price_cols or not ask_price_cols:
        raise ValueError("Order book dataframe missing required columns")

    bid_depth = df[bid_vol_cols].sum(axis=1)
    ask_depth = df[ask_vol_cols].sum(axis=1)
    total_depth = bid_depth + ask_depth
    depth_imbalance = (bid_depth - ask_depth) / total_depth.replace(0, np.nan)

    # Volume weighted spread across available levels
    bid_prices = df[bid_price_cols].to_numpy()
    ask_prices = df[ask_price_cols].to_numpy()
    bid_vols = df[bid_vol_cols].to_numpy()
    ask_vols = df[ask_vol_cols].to_numpy()
    spreads = ask_prices - bid_prices
    volumes = bid_vols + ask_vols
    vw_spread = np.sum(spreads * volumes, axis=1) / volumes.sum(axis=1)

    market_impact = vw_spread * depth_imbalance

    df = df.copy()
    df["depth_imbalance"] = depth_imbalance.fillna(0.0)
    df["vw_spread"] = vw_spread
    df["market_impact"] = market_impact.fillna(0.0)
    df["total_depth"] = total_depth
    # slippage approximated as spread plus absolute market impact
    df["slippage"] = np.abs(df["vw_spread"]) + np.abs(df["market_impact"])
    df["liquidity"] = total_depth
    return df


__all__ = ["load_order_book", "compute_order_book_features"]
