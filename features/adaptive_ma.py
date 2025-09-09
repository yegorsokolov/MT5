"""Kaufman's Adaptive Moving Average (KAMA)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _kama(
    series: pd.Series, period: int = 10, fast: int = 2, slow: int = 30
) -> pd.Series:
    change = series.diff(period).abs()
    volatility = series.diff().abs().rolling(period).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    kama = pd.Series(index=series.index, dtype=float)
    kama.iloc[:period] = series.iloc[:period]
    for i in range(period, len(series)):
        kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (
            series.iloc[i] - kama.iloc[i - 1]
        )
    return kama


def compute(
    df: pd.DataFrame, period: int = 10, fast: int = 2, slow: int = 30
) -> pd.DataFrame:
    """Compute Kaufman's Adaptive Moving Average and cross signals."""
    df = df.copy()
    price = df["Close"]
    kama = _kama(price, period, fast, slow)
    df["kama"] = kama
    cross = ((price > kama) & (price.shift(1) <= kama.shift(1))).astype(int) - (
        (price < kama) & (price.shift(1) >= kama.shift(1))
    ).astype(int)
    df["kama_cross"] = cross.fillna(0).astype(int)
    return df


__all__ = ["compute"]
