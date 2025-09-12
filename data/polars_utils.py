"""Helpers for converting between pandas and Polars DataFrames."""

from __future__ import annotations

from typing import Any

import pandas as pd

try:  # pragma: no cover - Polars is optional
    import polars as pl
except Exception:  # pragma: no cover
    pl = None  # type: ignore


def to_polars_df(df: Any):
    """Convert ``df`` to a :class:`polars.DataFrame`.

    Parameters
    ----------
    df:
        Input object which may already be a Polars frame or a pandas
        DataFrame.

    Returns
    -------
    polars.DataFrame
        Converted DataFrame.  Raises :class:`ImportError` if Polars is
        unavailable.
    """

    if pl is None:
        raise ImportError("polars is required but not installed")
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    raise TypeError("Unsupported dataframe type: %s" % type(df))


def to_pandas_df(df: Any) -> pd.DataFrame:
    """Convert ``df`` to :class:`pandas.DataFrame`.

    Parameters
    ----------
    df:
        Input object which may already be a pandas frame or a Polars frame.

    Returns
    -------
    pandas.DataFrame
        Converted DataFrame.  Raises :class:`ImportError` if Polars is
        unavailable and ``df`` is a Polars frame.
    """

    if isinstance(df, pd.DataFrame):
        return df
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.to_pandas()
    raise TypeError("Unsupported dataframe type: %s" % type(df))


__all__ = ["to_polars_df", "to_pandas_df"]
