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

    Parameters
    ----------
    data : Sequence[float] or pandas.Series
        Input values for which to compute the average.
    period : int
        Number of periods to include in the average.

    Returns
    -------
    float or pandas.Series
        When ``data`` is a Series the full rolling average is returned,
        otherwise only the most recent value is returned.
    """
    series = _to_series(data)
    result = series.rolling(period).mean()
    if isinstance(data, pd.Series):
        return result
    return float(result.iloc[-1])


def ema(data: Sequence[float] | pd.Series, period: int) -> float | pd.Series:
    """Exponential moving average.

    Parameters
    ----------
    data : Sequence[float] or pandas.Series
        Input values for which to compute the average.
    period : int
        Span of the exponential window.

    Returns
    -------
    float or pandas.Series
        The exponential moving average.  A Series is returned when the
        input is a Series; otherwise the latest value is returned.
    """
    series = _to_series(data)
    result = series.ewm(span=period, adjust=False).mean()
    if isinstance(data, pd.Series):
        return result
    return float(result.iloc[-1])


def rsi(data: Sequence[float] | pd.Series, period: int = 14) -> float | pd.Series:
    """Relative strength index.

    Parameters
    ----------
    data : Sequence[float] or pandas.Series
        Price or value series.
    period : int, default 14
        Number of periods for computing average gains and losses.

    Returns
    -------
    float or pandas.Series
        RSI values corresponding to ``data``.  When ``data`` is a Series the
        entire RSI series is returned, otherwise only the latest value is
        provided.
    """
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

    Parameters
    ----------
    data : Sequence[float] or pandas.Series
        Input data.
    period : int, default 20
        Rolling window size for the moving average.
    num_std : float, default 2.0
        Number of standard deviations for the band width.

    Returns
    -------
    tuple of float or pandas.Series
        ``(ma, upper, lower)`` values.  Each element is a Series when
        ``data`` is a Series, otherwise only the most recent float is returned.
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
    """Average true range.

    Parameters
    ----------
    high, low, close : Sequence[float] or pandas.Series
        High, low and close price data.
    period : int, default 14
        Window size for the moving average of the true range.

    Returns
    -------
    float or pandas.Series
        The ATR values.  When any input is a Series the full ATR series is
        returned; otherwise only the latest value is provided.
    """
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

    Parameters
    ----------
    data : Sequence[float] or pandas.Series
        Input data for the MACD calculation.
    fast : int, default 12
        Period for the fast EMA.
    slow : int, default 26
        Period for the slow EMA.
    signal : int, default 9
        Period for the signal line EMA.

    Returns
    -------
    tuple of float or pandas.Series
        ``(macd_line, signal_line, hist)``.  Elements are Series when the
        input is a Series; otherwise the latest float values are returned.
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
    """Stochastic oscillator.

    Parameters
    ----------
    high, low, close : Sequence[float] or pandas.Series
        High, low and close price data.
    k_period : int, default 14
        Number of periods for the %K calculation.
    d_period : int, default 3
        Window size for the %D moving average.

    Returns
    -------
    tuple of float or pandas.Series
        ``(%K, %D)`` values where ``%D`` is the moving average of ``%K``.  Each
        element is a Series when any input is a Series; otherwise only the
        latest floats are returned.
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


def vwap(
    price: Sequence[float] | pd.Series,
    volume: Sequence[float] | pd.Series,
    group: Sequence[object] | pd.Series | None = None,
) -> float | pd.Series:
    """Volume weighted average price.

    Parameters
    ----------
    price, volume:
        Price and volume data.  When Series are provided, the indices are
        aligned prior to computation.
    group:
        Optional grouping labels used to compute grouped cumulative VWAP,
        e.g. session or day buckets.  When omitted, the cumulative VWAP over
        the entire series is returned.

    Returns
    -------
    float or pandas.Series
        VWAP values aligned to ``price``.  A scalar is returned when plain
        sequences are supplied for all inputs, otherwise a Series is
        produced.
    """

    price_s = _to_series(price)
    volume_s = _to_series(volume)
    if len(price_s) != len(volume_s):
        raise ValueError("price and volume must share the same length")

    if group is None:
        pv = (price_s * volume_s).cumsum()
        vv = volume_s.cumsum()
    else:
        if isinstance(group, pd.Series):
            group_s = group
            if not group_s.index.equals(price_s.index):
                group_s = group_s.reindex(price_s.index)
        else:
            labels = list(group)
            if len(labels) != len(price_s):
                raise ValueError("group labels must match price length")
            group_s = pd.Series(labels, index=price_s.index)
        pv = (price_s * volume_s).groupby(group_s).cumsum()
        vv = volume_s.groupby(group_s).cumsum()

    result = pv / vv.replace(0, np.nan)
    if isinstance(price, pd.Series) or isinstance(volume, pd.Series) or isinstance(group, pd.Series):
        return result
    return float(result.iloc[-1])


__all__ = [
    "rsi",
    "atr",
    "bollinger",
    "sma",
    "ema",
    "macd",
    "stochastic",
    "vwap",
]
