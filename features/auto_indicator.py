"""Automatic indicator generation using basic transformations.

This module scans numeric columns and creates additional technical
indicators such as lagged values and rolling statistics.  It serves as a
light‑weight example of a self‑improving feature generator.  The
``compute`` function is intentionally simple so it can run quickly in
unit tests.  It can be extended with more sophisticated feature search
strategies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _generate_lags(series: pd.Series, lags: int = 3) -> pd.DataFrame:
    """Return DataFrame of lagged versions of ``series`` up to ``lags``."""
    data = {f"{series.name}_lag{lag}": series.shift(lag) for lag in range(1, lags + 1)}
    return pd.DataFrame(data)


def _rolling_stats(series: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Return rolling mean and std for ``series`` over provided windows."""
    frames = {}
    for win in windows:
        roll = series.rolling(win)
        frames[f"{series.name}_mean{win}"] = roll.mean()
        frames[f"{series.name}_std{win}"] = roll.std()
    return pd.DataFrame(frames)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional indicators from all numeric columns.

    The function automatically generates lagged values and simple rolling
    statistics for each numeric column.  Non‑numeric columns are ignored.
    The original dataframe is not modified; a copy with new features is
    returned.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = df[col]
        df = df.join(_generate_lags(series))
        df = df.join(_rolling_stats(series, windows=[5, 10]))
    df = df.dropna(axis=1, how="all")
    return df


__all__ = ["compute"]
