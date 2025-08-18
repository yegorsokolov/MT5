from __future__ import annotations

"""Data quality utilities for detecting gaps and outliers in tick data.

This module provides simple routines for identifying time gaps, removing
statistical outliers using z-score and median based filters and filling in
missing values via interpolation.  The functions are intentionally lightweight
so they can be executed in both offline history loading and real-time data
pipelines.
"""

import logging
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _to_timedelta(value: str | int | float | pd.Timedelta) -> pd.Timedelta:
    """Return ``pd.Timedelta`` for ``value``.

    Parameters
    ----------
    value:
        Either a string like ``"1s"`` or numeric seconds.  ``pd.Timedelta``
        instances are returned as-is.
    """

    if isinstance(value, pd.Timedelta):
        return value
    if isinstance(value, (int, float)):
        return pd.Timedelta(seconds=float(value))
    return pd.Timedelta(value)


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

def detect_gaps(
    df: pd.DataFrame,
    *,
    ts_col: str = "Timestamp",
    max_gap: str | int | float | pd.Timedelta = "1s",
) -> Tuple[pd.Series, int]:
    """Return boolean mask of gaps and the gap count.

    A gap is defined as the difference between consecutive timestamps being
    greater than ``max_gap``.
    """

    if df.empty:
        return pd.Series(dtype=bool), 0

    gap = _to_timedelta(max_gap)
    ts = pd.to_datetime(df[ts_col])
    diff = ts.diff()
    mask = diff > gap
    count = int(mask.sum())
    if count:
        logger.warning("Detected %d gaps greater than %s", count, gap)
    return mask, count


# ---------------------------------------------------------------------------
# Z-score based outlier removal
# ---------------------------------------------------------------------------

def zscore_filter(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    threshold: float = 3.0,
) -> Dict[str, int]:
    """Replace values with NaN where the z-score exceeds ``threshold``.

    Returns a dictionary mapping column names to number of outliers replaced.
    """

    out: Dict[str, int] = {}
    for col in cols:
        series = df[col].astype(float)
        std = series.std(ddof=0)
        if std == 0 or series.empty:
            out[col] = 0
            continue
        z = (series - series.mean()) / std
        mask = z.abs() > threshold
        out[col] = int(mask.sum())
        if out[col]:
            logger.warning("Z-score filter removed %d outliers from %s", out[col], col)
            df.loc[mask, col] = np.nan
    return out


# ---------------------------------------------------------------------------
# Median absolute deviation based filter
# ---------------------------------------------------------------------------

def median_filter(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    window: int = 5,
    threshold: float = 5.0,
) -> Dict[str, int]:
    """Replace values deviating from rolling median by ``threshold`` * MAD.

    Parameters
    ----------
    df:
        Input dataframe.
    cols:
        Columns to filter.
    window:
        Rolling window size for computing median and MAD.
    threshold:
        Multiplicative factor for MAD to mark outliers.
    """

    out: Dict[str, int] = {}
    for col in cols:
        series = df[col].astype(float)
        med = series.rolling(window, center=True, min_periods=1).median()
        mad = (series - med).abs().rolling(window, center=True, min_periods=1).median()
        diff = (series - med).abs()
        mask = diff > threshold * mad.replace(0, np.nan)
        out[col] = int(mask.sum())
        if out[col]:
            logger.warning("Median filter removed %d outliers from %s", out[col], col)
            df.loc[mask, col] = np.nan
    return out


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def interpolate_gaps(
    df: pd.DataFrame,
    *,
    ts_col: str = "Timestamp",
    cols: Sequence[str] | None = None,
    freq: str | int | float | pd.Timedelta = "1s",
) -> pd.DataFrame:
    """Insert missing timestamps and interpolate numeric columns.

    Parameters
    ----------
    df:
        Input dataframe.
    ts_col:
        Timestamp column name.
    cols:
        Columns to interpolate.  Defaults to all non-timestamp columns.
    freq:
        Frequency for reindexing.  If numeric, interpreted as seconds.
    """

    if df.empty:
        return df

    freq_td = _to_timedelta(freq)
    cols = list(cols or df.columns.difference([ts_col]))
    result = df.copy()
    result[ts_col] = pd.to_datetime(result[ts_col])
    result.set_index(ts_col, inplace=True)
    full_index = pd.date_range(result.index.min(), result.index.max(), freq=freq_td)
    result = result.reindex(full_index)
    result[cols] = result[cols].interpolate().ffill().bfill()
    result[ts_col] = result.index
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def apply_quality_checks(
    df: pd.DataFrame,
    *,
    ts_col: str = "Timestamp",
    cols: Sequence[str] = ("Bid", "Ask"),
    max_gap: str | int | float | pd.Timedelta = "1s",
    z_threshold: float = 3.0,
    med_window: int = 5,
    med_threshold: float = 5.0,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Apply gap detection, outlier filters and interpolation.

    Returns the cleaned dataframe and a report dict containing counts of
    detected issues.
    """

    frame = df.copy()
    cols = list(cols)
    _, gap_count = detect_gaps(frame, ts_col=ts_col, max_gap=max_gap)
    if gap_count:
        frame = interpolate_gaps(frame, ts_col=ts_col, cols=cols, freq=max_gap)
    z_out = zscore_filter(frame, cols, threshold=z_threshold)
    m_out = median_filter(frame, cols, window=med_window, threshold=med_threshold)
    if frame[cols].isna().any().any():
        frame[cols] = frame[cols].interpolate().ffill().bfill()
    report = {
        "gaps": gap_count,
        "zscore": int(sum(z_out.values())),
        "median": int(sum(m_out.values())),
    }
    return frame, report


__all__ = [
    "detect_gaps",
    "zscore_filter",
    "median_filter",
    "interpolate_gaps",
    "apply_quality_checks",
]
