"""SuperTrend indicator computed from ATR."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> pd.DataFrame:
    """Compute SuperTrend line and breakout signals."""
    df = df.copy()
    high = df.get("High", df["Close"])
    low = df.get("Low", df["Close"])
    close = df["Close"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(period).mean()

    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    supertrend = pd.Series(np.nan, index=df.index)
    direction = 0
    for i in range(period, len(df)):
        if direction <= 0 and close.iloc[i] > upper.iloc[i - 1]:
            direction = 1
        elif direction >= 0 and close.iloc[i] < lower.iloc[i - 1]:
            direction = -1
        supertrend.iloc[i] = lower.iloc[i] if direction == 1 else upper.iloc[i]

    df["supertrend"] = supertrend
    break_signal = (
        (close > supertrend) & (close.shift(1) <= supertrend.shift(1))
    ).astype(int) - (
        (close < supertrend) & (close.shift(1) >= supertrend.shift(1))
    ).astype(
        int
    )
    df["supertrend_break"] = break_signal.fillna(0).astype(int)
    return df


__all__ = ["compute"]
