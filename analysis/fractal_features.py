"""Fractal based feature computations.

This module provides utilities to compute the rolling Hurst exponent and
fractal dimension of a 1D time series.  The implementation uses the Katz
fractal dimension which is inexpensive and stable for short sequences.
The Hurst exponent is derived as the inverse of the fractal dimension,
providing values close to ``1`` for persistent trends and around ``0.5``
for random walks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _katz_fd(arr: np.ndarray) -> float:
    """Compute Katz fractal dimension for a sequence.

    Parameters
    ----------
    arr : np.ndarray
        Input 1D array.

    Returns
    -------
    float
        Estimated fractal dimension.
    """
    arr = np.asarray(arr, dtype=float)
    n = arr.size
    if n < 2:
        return np.nan
    L = np.sum(np.abs(np.diff(arr)))
    d = np.max(np.abs(arr - arr[0]))
    if L == 0 or d == 0:
        return 1.0
    return np.log10(n) / (np.log10(n) + np.log10(d / L))


def rolling_fractal_features(
    series: pd.Series | np.ndarray,
    window: int = 128,
) -> pd.DataFrame:
    """Compute rolling Hurst exponent and fractal dimension.

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Input signal.
    window : int, default 128
        Size of the rolling window.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``hurst`` and ``fractal_dim`` containing the
        rolling estimates.
    """
    if isinstance(series, pd.Series):
        index = series.index
        arr = series.to_numpy()
    else:
        arr = np.asarray(series)
        index = None

    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    hurst = np.full(n, np.nan, dtype=float)
    fd = np.full(n, np.nan, dtype=float)

    for end in range(window, n + 1):
        segment = arr[end - window : end]
        fdim = _katz_fd(segment)
        fd[end - 1] = fdim
        if fdim != 0 and not np.isnan(fdim):
            hurst[end - 1] = 1.0 / fdim
        else:
            hurst[end - 1] = np.nan

    data = {"hurst": hurst, "fractal_dim": fd}
    return pd.DataFrame(data, index=index)


def rolling_hurst_exponent(
    series: pd.Series | np.ndarray, window: int = 128
) -> pd.Series:
    """Rolling Hurst exponent of ``series``.

    This is a thin wrapper around :func:`rolling_fractal_features` returning
    only the ``hurst`` column.
    """
    return rolling_fractal_features(series, window=window)["hurst"]


def rolling_fractal_dimension(
    series: pd.Series | np.ndarray, window: int = 128
) -> pd.Series:
    """Rolling fractal dimension of ``series``.

    This is a thin wrapper around :func:`rolling_fractal_features` returning
    only the ``fractal_dim`` column.
    """
    return rolling_fractal_features(series, window=window)["fractal_dim"]
