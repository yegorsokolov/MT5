"""Utilities for aggregating tick data across multiple timeframes."""

from __future__ import annotations

import pandas as pd


def _tf_label(tf: str) -> str:
    """Convert a pandas resample rule into a compact label.

    Examples
    --------
    >>> _tf_label("15min")
    '15m'
    >>> _tf_label("1H")
    '1h'
    """

    td = pd.to_timedelta(tf)
    total_minutes = int(td.total_seconds() // 60)
    if total_minutes % (24 * 60) == 0:
        days = total_minutes // (24 * 60)
        return f"{days}d"
    if total_minutes % 60 == 0:
        hours = total_minutes // 60
        return f"{hours}h"
    return f"{total_minutes}m"


def aggregate_timeframes(df: pd.DataFrame, timeframes: list[str]) -> pd.DataFrame:
    """Resample numeric columns over multiple timeframes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least a ``Timestamp`` column.
    timeframes : list[str]
        List of pandas resample strings, e.g. ``['1min', '15min', '1H']``.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``Timestamp`` and aggregated columns for each timeframe.
    """

    if "Timestamp" not in df.columns:
        raise ValueError("Timestamp column is required")

    df_indexed = df.set_index(pd.to_datetime(df["Timestamp"]))
    numeric_cols = df_indexed.select_dtypes(include="number").columns

    out = pd.DataFrame(index=df_indexed.index)
    for tf in timeframes:
        label = _tf_label(tf)
        resampled = df_indexed[numeric_cols].resample(tf).agg(["mean", "std"])
        resampled.columns = [f"{col}_{label}_{stat}" for col, stat in resampled.columns]
        resampled = resampled.reindex(df_indexed.index, method="ffill")
        out = pd.concat([out, resampled], axis=1)

    out = out.reset_index().rename(columns={"index": "Timestamp"})
    return out

__all__ = ["aggregate_timeframes"]
