"""Higher timeframe technical indicators via resampling.

This module derives moving averages and RSI values on higher timeframes
(e.g. hourly, 4-hour) from a lower timeframe price series.  It resamples
the input dataframe by ``Timestamp`` and forward-fills the resulting
indicators back to the original frequency so they can be merged with
other feature pipelines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute(
    df: pd.DataFrame,
    timeframes: tuple[str, ...] = ("1h", "4h"),
    ma_window: int = 20,
    rsi_window: int = 14,
) -> pd.DataFrame:
    """Compute higher timeframe moving averages and RSI.

    Parameters
    ----------
    df:
        Dataframe containing a ``Timestamp`` column and either ``mid`` or
        ``Ask``/``Bid`` prices.  ``mid`` will be computed if absent.
    timeframes:
        Resample intervals, e.g. ``("1H", "4H")``.
    ma_window:
        Window size for moving average on the resampled series.
    rsi_window:
        Window size for RSI on the resampled series.

    Returns
    -------
    pandas.DataFrame
        Original dataframe with additional ``ma_<tf>`` and ``rsi_<tf>``
        columns for each timeframe.
    """

    df = df.copy()
    if "mid" not in df.columns:
        if {"Ask", "Bid"}.issubset(df.columns):
            df["mid"] = (df["Ask"] + df["Bid"]) / 2
        else:
            raise KeyError("DataFrame must contain 'mid' or 'Ask'/'Bid' columns")

    if "Timestamp" not in df.columns:
        raise KeyError("DataFrame must contain 'Timestamp' column")

    ts = pd.to_datetime(df["Timestamp"])
    df = df.set_index(ts)

    price = df["mid"]
    for tf in timeframes:
        resampled = price.resample(tf).last()
        ma = resampled.rolling(ma_window).mean()
        rsi = _rsi(resampled, rsi_window)
        tf_key = tf.lower()
        df[f"ma_{tf_key}"] = ma.reindex(df.index, method="ffill")
        df[f"rsi_{tf_key}"] = rsi.reindex(df.index, method="ffill")

    return df.reset_index(drop=True)


__all__ = ["compute"]
