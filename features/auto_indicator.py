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
from typing import Iterable, Sequence, Optional, Union


def _generate_lags(series: pd.Series, lags: Sequence[int]) -> pd.DataFrame:
    """Return DataFrame of lagged versions of ``series``."""
    data = {f"{series.name}_lag{lag}": series.shift(lag) for lag in lags}
    return pd.DataFrame(data)


def _rolling_stats(series: pd.Series, windows: Sequence[int]) -> pd.DataFrame:
    """Return rolling mean and std for ``series`` over provided windows."""
    frames = {}
    for win in windows:
        roll = series.rolling(win)
        frames[f"{series.name}_mean{win}"] = roll.mean()
        frames[f"{series.name}_std{win}"] = roll.std()
    return pd.DataFrame(frames)


def compute(
    df: pd.DataFrame,
    lags: Union[int, Sequence[int]] = 3,
    windows: Sequence[int] = (5, 10),
    skip: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Create additional indicators from numeric columns.

    Parameters
    ----------
    df:
        Input dataframe.
    lags:
        Either an integer specifying the number of sequential lags starting
        at 1 or an explicit iterable of lag steps.
    windows:
        Rolling window sizes for which mean and standard deviation are
        calculated.
    skip:
        Optional iterable of column names to exclude from transformation,
        e.g. target columns to avoid data leakage.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if skip:
        skip_set = set(skip)
        numeric_cols = [c for c in numeric_cols if c not in skip_set]
    lag_seq = range(1, lags + 1) if isinstance(lags, int) else list(lags)
    win_seq = list(windows)
    for col in numeric_cols:
        series = df[col]
        df = df.join(_generate_lags(series, lag_seq))
        df = df.join(_rolling_stats(series, win_seq))
    df = df.dropna(axis=1, how="all")
    return df


__all__ = ["compute"]
