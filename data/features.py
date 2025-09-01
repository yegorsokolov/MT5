"""Lightweight feature engineering orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from features import get_feature_pipeline
from features.news import (
    add_economic_calendar_features,
    add_news_sentiment_features,
)
from features.cross_asset import (
    add_index_features,
    add_cross_asset_features,
)

logger = logging.getLogger(__name__)


def make_features(df: pd.DataFrame, validate: bool = False) -> pd.DataFrame:
    """Generate model features by executing registered modules sequentially."""
    for compute in get_feature_pipeline():
        df = compute(df)
    adjacency = df.attrs.get("adjacency_matrices")
    try:
        import dataset  # type: ignore

        plugins = getattr(dataset, "FEATURE_PLUGINS", [])
    except Exception:
        plugins = []
    for plugin in plugins:
        df = plugin(df, adjacency_matrices=adjacency)
    return df


def make_features_memmap(path: str | Path, chunk_size: int = 1000) -> pd.DataFrame:
    """Load history from ``path`` and compute features."""
    df = pd.read_parquet(path)
    return make_features(df, validate=False)


# -- Technical helpers -------------------------------------------------

def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def ma_cross_signal(df: pd.DataFrame, short: str = "ma_10", long: str = "ma_30") -> pd.Series:
    cross_up = (df[short] > df[long]) & (df[short].shift(1) <= df[long].shift(1))
    cross_down = (df[short] < df[long]) & (df[short].shift(1) >= df[long].shift(1))
    signal = pd.Series(0, index=df.index)
    signal[cross_up] = 1
    signal[cross_down] = -1
    return signal


def train_test_split(df: pd.DataFrame, n_train: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "Symbol" in df.columns:
        trains: List[pd.DataFrame] = []
        tests: List[pd.DataFrame] = []
        for _, g in df.groupby("Symbol"):
            trains.append(g.iloc[:n_train].copy())
            tests.append(g.iloc[n_train:].copy())
        return pd.concat(trains, ignore_index=True), pd.concat(tests, ignore_index=True)
    train = df.iloc[:n_train].copy()
    test = df.iloc[n_train:].copy()
    return train, test


def make_sequence_arrays(
    df: pd.DataFrame, features: List[str], seq_len: int, label_col: str = "return"
) -> Tuple[np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    groups = [df]
    if "Symbol" in df.columns:
        groups = [g for _, g in df.groupby("Symbol")]
    for g in groups:
        values = g[features].values
        if label_col == "return":
            targets = (g["return"].shift(-1) > 0).astype(int).values
            limit = len(g) - 1
        else:
            targets = g[label_col].values
            limit = len(g)
        for i in range(seq_len, limit):
            X_list.append(values[i - seq_len : i])
            y_list.append(targets[i])
    X = np.stack(X_list)
    y = np.array(y_list)
    return X, y


__all__ = [
    "add_index_features",
    "add_economic_calendar_features",
    "add_news_sentiment_features",
    "add_cross_asset_features",
    "make_features",
    "make_features_memmap",
    "compute_rsi",
    "ma_cross_signal",
    "train_test_split",
    "make_sequence_arrays",
]
