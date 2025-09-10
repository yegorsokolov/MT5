"""Keltner Channel squeeze breakout features."""

from __future__ import annotations

import pandas as pd


def compute(
    df: "pd.DataFrame",
    window: int = 20,
    bb_mult: float = 2.0,
    kc_mult: float = 1.5,
) -> "pd.DataFrame":
    """Compute Keltner Channels, Bollinger Band width and squeeze breakout.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with ``Close`` prices and optional ``High``/``Low``.
    window : int, optional
        Lookback period for moving averages and ATR, by default 20.
    bb_mult : float, optional
        Bollinger Band standard deviation multiplier, by default 2.0.
    kc_mult : float, optional
        Keltner Channel ATR multiplier, by default 1.5.

    Returns
    -------
    pd.DataFrame
        DataFrame with added ``kc_upper``, ``kc_lower``, ``bb_width`` and
        ``squeeze_break`` columns.
    """

    df = df.copy()
    close = df["Close"]
    high = df.get("High", close)
    low = df.get("Low", close)

    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    bb_upper = ma + bb_mult * std
    bb_lower = ma - bb_mult * std
    df["bb_width"] = (bb_upper - bb_lower) / ma

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(window).mean()
    ema = close.ewm(span=window, adjust=False).mean()
    kc_upper = ema + kc_mult * atr
    kc_lower = ema - kc_mult * atr
    df["kc_upper"] = kc_upper
    df["kc_lower"] = kc_lower

    squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    break_signal = pd.Series(0, index=df.index)
    broke_up = squeeze_on.shift(1) & ~squeeze_on & (close > bb_upper)
    broke_down = squeeze_on.shift(1) & ~squeeze_on & (close < bb_lower)
    break_signal[broke_up] = 1
    break_signal[broke_down] = -1
    df["squeeze_break"] = break_signal.fillna(0).astype(int)

    return df


__all__ = ["compute"]
