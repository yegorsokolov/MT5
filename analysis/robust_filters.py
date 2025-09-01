from __future__ import annotations

"""Robust filtering utilities for anomaly mitigation.

This module provides simple median and trimmed mean filters along with a
z-score based clamp. These filters can be combined to clean noisy market
streams prior to feature engineering.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def median_filter(series: pd.Series, window: int = 5) -> pd.Series:
    """Apply a rolling median filter.

    Parameters
    ----------
    series : pd.Series
        Input series to smooth.
    window : int, optional
        Rolling window size, by default 5.
    """

    return series.rolling(window, center=True, min_periods=1).median()


def trimmed_mean_filter(
    series: pd.Series, window: int = 5, trim_ratio: float = 0.1
) -> pd.Series:
    """Apply a rolling trimmed-mean filter.

    Parameters
    ----------
    series : pd.Series
        Input series to smooth.
    window : int, optional
        Rolling window size, by default 5.
    trim_ratio : float, optional
        Fraction to trim from each tail, by default 0.1.
    """

    def _trimmed(arr: np.ndarray) -> float:
        if len(arr) == 0:
            return np.nan
        k = int(len(arr) * trim_ratio)
        if k == 0:
            return float(np.mean(arr))
        trimmed = np.sort(arr)[k:-k] if len(arr) > 2 * k else np.sort(arr)[k:]
        return float(np.mean(trimmed)) if len(trimmed) else float(np.mean(arr))

    return series.rolling(window, center=True, min_periods=1).apply(
        _trimmed, raw=True
    )


def zscore_clamp(series: pd.Series, threshold: float = 3.0) -> Tuple[pd.Series, List[int]]:
    """Clamp values exceeding a z-score threshold.

    Parameters
    ----------
    series : pd.Series
        Input series.
    threshold : float, optional
        Z-score threshold, by default 3.0.

    Returns
    -------
    Tuple[pd.Series, List[int]]
        Clamped series and list of indices where clamping occurred.
    """

    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return series, []
    zscores = (series - mean) / std
    mask = zscores.abs() > threshold
    clamped = series.copy()
    clamped[mask] = mean + np.clip(zscores[mask], -threshold, threshold) * std
    return clamped, series[mask].index.tolist()


def robust_preprocess(
    df: pd.DataFrame,
    window: int = 5,
    trim_ratio: float = 0.1,
    z_thresh: float = 3.0,
) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    """Apply robust filters to all numeric columns of ``df``.

    Returns a cleaned dataframe and a mapping of column names to indices of
    detected anomalies.
    """

    out = df.copy()
    anomalies: Dict[str, List[int]] = {}
    for col in out.select_dtypes(include=[np.number]).columns:
        filtered = median_filter(out[col], window)
        filtered = trimmed_mean_filter(filtered, window, trim_ratio)
        clamped, idx = zscore_clamp(filtered, z_thresh)
        out[col] = clamped
        anomalies[col] = idx
    return out, anomalies
