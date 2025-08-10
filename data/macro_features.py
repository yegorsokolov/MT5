"""Utilities for loading macroeconomic indicator series.

This module provides :func:`load_macro_series` which fetches macroeconomic
indicators such as GDP growth, CPI and interest rates from either local CSV
files or external APIs.  The function returns a dataframe suitable for merging
with price data on the ``Date`` column.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def _read_local_csv(symbol: str) -> pd.DataFrame:
    """Attempt to read a macro series from common local paths.

    Parameters
    ----------
    symbol : str
        Identifier of the macro series.  Files are looked for under
        ``data/`` and ``dataset/`` with ``.csv`` extension.

    Returns
    -------
    pd.DataFrame
        Dataframe containing at least ``Date`` and the value column, or an
        empty dataframe if the file is not found or cannot be read.
    """

    paths = [Path("data") / f"{symbol}.csv", Path("dataset") / f"{symbol}.csv"]
    for path in paths:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:  # pragma: no cover - local file read failure
                logger.warning("Failed to load macro series %s from %s", symbol, path)
                return pd.DataFrame()
    return pd.DataFrame()


def load_macro_series(symbols: List[str]) -> pd.DataFrame:
    """Load macroeconomic time series for the given symbols.

    The function first searches for local CSV files.  If a series is not
    available locally it attempts to fetch the data via ``pandas_datareader``
    from public APIs such as FRED.  Any series that cannot be retrieved is
    skipped.

    Parameters
    ----------
    symbols : list[str]
        List of series identifiers to load.

    Returns
    -------
    pd.DataFrame
        Dataframe with a ``Date`` column and one column per successfully
        retrieved symbol.  The dataframe is sorted by date and missing series
        result in an empty column.
    """

    frames: list[pd.DataFrame] = []
    for sym in symbols:
        df = _read_local_csv(sym)
        if df.empty:
            try:  # pragma: no cover - network access may not be available
                from pandas_datareader import data as web  # type: ignore

                fetched = web.DataReader(sym, "fred")
                fetched = fetched.rename_axis("Date").reset_index()
                df = fetched
            except Exception:
                logger.debug("Unable to fetch macro series %s", sym)
                continue
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        value_col = [c for c in df.columns if c != "Date"][0]
        frames.append(df[["Date", value_col]].rename(columns={value_col: sym}))

    if not frames:
        # Return empty dataframe with expected columns for downstream merging
        return pd.DataFrame(columns=["Date"] + list(symbols))

    out = frames[0]
    for extra in frames[1:]:
        out = out.merge(extra, on="Date", how="outer")
    return out.sort_values("Date")


__all__ = ["load_macro_series"]
