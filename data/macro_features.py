"""Utilities for loading macroeconomic indicators.

The repository historically exposed :func:`load_macro_series` which simply
returned stand‑alone time series.  For the learning pipeline we now require a
convenience helper that merges any available macro datasets with price history
on a per date/region basis.  The new :func:`load_macro_features` takes a price
dataframe and enriches it with columns prefixed by ``"macro_"``.  Local CSV
files placed under ``dataset/macro`` or ``data/macro`` are automatically
discovered and merged using a backward ``merge_asof`` join.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

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


def _load_all_local() -> pd.DataFrame:
    """Load all available macro CSVs from ``dataset/macro`` and ``data/macro``.

    Files are expected to contain at least ``Date`` and ``Region`` columns plus
    one or more value columns.  All discovered series are outer merged on date
    and region.
    """

    frames: List[pd.DataFrame] = []
    for base in [Path("dataset") / "macro", Path("data") / "macro"]:
        if not base.exists():
            continue
        for path in base.glob("*.csv"):
            try:
                df = pd.read_csv(path)
            except Exception:  # pragma: no cover - local read failure
                logger.warning("Failed to read macro data from %s", path)
                continue
            if "Date" not in df.columns:
                df = df.rename(columns={df.columns[0]: "Date"})
            if "Region" not in df.columns:
                # Derive region from filename if not provided
                df["Region"] = path.stem
            df["Date"] = pd.to_datetime(df["Date"], utc=True)
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["Date", "Region"])

    out = frames[0]
    for extra in frames[1:]:
        out = out.merge(extra, on=["Date", "Region"], how="outer")
    return out.sort_values(["Region", "Date"])


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


def load_macro_features(
    df: pd.DataFrame, macro_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Merge macroeconomic indicators into ``df``.

    Parameters
    ----------
    df:
        Price dataframe containing a ``Timestamp`` column and optionally a
        ``Region`` column.  Macro features will be aligned on timestamp and
        region using a backward fill.
    macro_df:
        Optional dataframe containing macro series.  When ``None`` all
        available CSV files from ``dataset/macro`` and ``data/macro`` are
        loaded.

    Returns
    -------
    pd.DataFrame
        ``df`` enriched with ``macro_`` prefixed columns.  Missing values are
        forward filled then remaining gaps are set to ``0`` so downstream
        models can rely on their presence.
    """

    if macro_df is None:
        macro_df = _load_all_local()

    if macro_df.empty:
        # Ensure expected prefix columns exist if we can infer any names
        for col in [c for c in macro_df.columns if c not in {"Date", "Region"}]:
            df[f"macro_{col}"] = 0.0
        return df

    macro_df = macro_df.rename(columns={"Date": "macro_date"})
    macro_df = macro_df.sort_values(["Region", "macro_date"])

    merge_kwargs = {
        "left_on": "Timestamp",
        "right_on": "macro_date",
        "direction": "backward",
    }
    if "Region" in df.columns and "Region" in macro_df.columns:
        merge_kwargs["by"] = "Region"

    merged = pd.merge_asof(
        df.sort_values("Timestamp"), macro_df, **merge_kwargs
    ).drop(columns=["macro_date"])

    value_cols = [c for c in macro_df.columns if c not in {"macro_date", "Region"}]
    for col in value_cols:
        if col in merged.columns:
            merged[col] = merged[col].ffill().fillna(0.0)
            merged.rename(columns={col: f"macro_{col}"}, inplace=True)
        else:
            merged[f"macro_{col}"] = 0.0

    return merged


__all__ = ["load_macro_series", "load_macro_features"]
