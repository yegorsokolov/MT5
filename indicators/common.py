"""Shared technical indicator implementations."""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import pandas as pd


def _to_series(data: Sequence[float] | pd.Series) -> pd.Series:
    """Convert ``data`` to a pandas Series."""
    if isinstance(data, pd.Series):
        return data
    return pd.Series(list(data))


def sma(data: Sequence[float] | pd.Series, period: int) -> float | pd.Series:
    """Simple moving average.

    Returns a pandas Series when ``data`` is a Series otherwise the latest
    value as ``float``.
    """
    series = _to_series(data)
    result = series.rolling(period).mean()
    if isinstance(data, pd.Series):
        return result
    return float(result.iloc[-1])


def ema(data: Sequence[float] | pd.Series, period: int) -> float | pd.Series:
    """Exponential moving average."""
    series = _to_series(data)
    result = series.ewm(span=period, adjust=False).mean()
    if isinstance(data, pd.Series):
        return result
    return float(result.iloc[-1])


def rsi(data: Sequence[float] | pd.Series, period: int = 14) -> float | pd.Series:
    """Relative Strength Index."""
    series = _to_series(data)
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    if isinstance(data, pd.Series):
        return rsi_series
    return float(rsi_series.iloc[-1])


def bollinger(
    data: Sequence[float] | pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[float | pd.Series, float | pd.Series, float | pd.Series]:
    """Bollinger Bands.

    Returns ``(ma, upper, lower)``. For Series input the return values are
    Series objects; otherwise the latest values are returned as floats.
    """
    series = _to_series(data)
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    if isinstance(data, pd.Series):
        return ma, upper, lower
    return float(ma.iloc[-1]), float(upper.iloc[-1]), float(lower.iloc[-1])


def atr(
    high: Sequence[float] | pd.Series,
    low: Sequence[float] | pd.Series,
    close: Sequence[float] | pd.Series,
    period: int = 14,
) -> float | pd.Series:
    """Average True Range."""
    high_s = _to_series(high)
    low_s = _to_series(low)
    close_s = _to_series(close)
    prev_close = close_s.shift(1)
    tr = pd.concat(
        [high_s - low_s, (high_s - prev_close).abs(), (low_s - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_series = tr.rolling(period).mean()
    if (
        isinstance(high, pd.Series)
        or isinstance(low, pd.Series)
        or isinstance(close, pd.Series)
    ):
        return atr_series
    return float(atr_series.iloc[-1])


def macd(
    data: Sequence[float] | pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[float | pd.Series, float | pd.Series, float | pd.Series]:
    """Moving Average Convergence Divergence (MACD).

    Returns ``(macd_line, signal_line, hist)``. When ``data`` is a Series the
    return values are Series objects otherwise the latest values are returned as
    floats.
    """
    series = _to_series(data)
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    if isinstance(data, pd.Series):
        return macd_line, signal_line, hist
    return (
        float(macd_line.iloc[-1]),
        float(signal_line.iloc[-1]),
        float(hist.iloc[-1]),
    )


def stochastic(
    high: Sequence[float] | pd.Series,
    low: Sequence[float] | pd.Series,
    close: Sequence[float] | pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[float | pd.Series, float | pd.Series]:
    """Stochastic Oscillator.

    Returns ``(%K, %D)`` where ``%D`` is the moving average of ``%K``.
    For Series input the return values are Series objects; otherwise the
    latest values are returned as floats.
    """
    high_s = _to_series(high)
    low_s = _to_series(low)
    close_s = _to_series(close)
    lowest_low = low_s.rolling(k_period).min()
    highest_high = high_s.rolling(k_period).max()
    k = 100 * (close_s - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_period).mean()
    if (
        isinstance(high, pd.Series)
        or isinstance(low, pd.Series)
        or isinstance(close, pd.Series)
    ):
        return k, d
    return float(k.iloc[-1]), float(d.iloc[-1])


__all__ = [
    "rsi",
    "atr",
    "bollinger",
    "sma",
    "ema",
    "macd",
    "stochastic",
]
