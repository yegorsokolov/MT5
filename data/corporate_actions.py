"""Lightweight corporate actions data loaders.

This module ingests dividend payments, stock split ratios, insider
transaction summaries and 13F institutional holdings from either local CSV
files or optional user supplied fetchers.  The functions are intentionally
simple so they work in minimal test environments and avoid relying on
subscription APIs.  Dataframes returned by the loaders are aligned on
``Date`` and ``Symbol`` so they can be merged using
:func:`pandas.merge_asof`.  Callers may provide custom ``fetcher``
callables to override the default CSV-only behaviour when they have access
to alternative data sources.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV helpers

def _read_category_csv(symbol: str, category: str) -> pd.DataFrame:
    """Return a dataframe for ``symbol`` under ``corporate_actions/<category>``.

    The function searches both ``dataset`` and ``data`` directories and
    silently returns an empty dataframe when no file is found or reading fails.
    """

    paths = [
        Path("dataset") / "corporate_actions" / category / f"{symbol}.csv",
        Path("data") / "corporate_actions" / category / f"{symbol}.csv",
    ]
    for path in paths:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:  # pragma: no cover - local read failure
                logger.warning(
                    "Failed to read %s corporate action for %s from %s",
                    category,
                    symbol,
                    path,
                )
                return pd.DataFrame()
    return pd.DataFrame()


def load_dividends(
    symbols: Iterable[str], fetcher: Optional[Callable[[str], pd.DataFrame]] = None
) -> pd.DataFrame:
    """Return dividend data for ``symbols`` as ``Date``/``Symbol``/``dividend``."""

    frames: List[pd.DataFrame] = []
    for sym in symbols:
        df = _read_category_csv(sym, "dividends")
        if df.empty and fetcher is not None:
            try:
                df = fetcher(sym)
            except Exception:  # pragma: no cover - fetcher failure
                df = pd.DataFrame()
        if df.empty:
            continue
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        val_col = "dividend" if "dividend" in df.columns else df.columns[1]
        df = df[["Date", val_col]].rename(columns={val_col: "dividend"})
        df["Symbol"] = sym
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["Date", "Symbol", "dividend"])
    return pd.concat(frames, ignore_index=True).sort_values(["Symbol", "Date"])


def load_splits(
    symbols: Iterable[str], fetcher: Optional[Callable[[str], pd.DataFrame]] = None
) -> pd.DataFrame:
    """Return split data for ``symbols``."""

    frames: List[pd.DataFrame] = []
    for sym in symbols:
        df = _read_category_csv(sym, "splits")
        if df.empty and fetcher is not None:
            try:
                df = fetcher(sym)
            except Exception:  # pragma: no cover
                df = pd.DataFrame()
        if df.empty:
            continue
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        val_col = "split" if "split" in df.columns else df.columns[1]
        df = df[["Date", val_col]].rename(columns={val_col: "split"})
        df["Symbol"] = sym
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["Date", "Symbol", "split"])
    return pd.concat(frames, ignore_index=True).sort_values(["Symbol", "Date"])


def load_insider_filings(
    symbols: Iterable[str], fetcher: Optional[Callable[[str], pd.DataFrame]] = None
) -> pd.DataFrame:
    """Return insider trading aggregates for ``symbols``."""

    frames: List[pd.DataFrame] = []
    for sym in symbols:
        df = _read_category_csv(sym, "insider")
        if df.empty and fetcher is not None:
            try:
                df = fetcher(sym)
            except Exception:  # pragma: no cover
                df = pd.DataFrame()
        if df.empty:
            continue
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        val_col = (
            "insider_trades" if "insider_trades" in df.columns else df.columns[1]
        )
        df = df[["Date", val_col]].rename(columns={val_col: "insider_trades"})
        df["Symbol"] = sym
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["Date", "Symbol", "insider_trades"])
    return pd.concat(frames, ignore_index=True).sort_values(["Symbol", "Date"])


def load_13f_filings(
    symbols: Iterable[str], fetcher: Optional[Callable[[str], pd.DataFrame]] = None
) -> pd.DataFrame:
    """Return institutional holdings from 13F filings for ``symbols``."""

    frames: List[pd.DataFrame] = []
    for sym in symbols:
        df = _read_category_csv(sym, "thirteenf")
        if df.empty and fetcher is not None:
            try:
                df = fetcher(sym)
            except Exception:  # pragma: no cover
                df = pd.DataFrame()
        if df.empty:
            continue
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        val_col = (
            "institutional_holdings"
            if "institutional_holdings" in df.columns
            else df.columns[1]
        )
        df = df[["Date", val_col]].rename(
            columns={val_col: "institutional_holdings"}
        )
        df["Symbol"] = sym
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["Date", "Symbol", "institutional_holdings"])
    return pd.concat(frames, ignore_index=True).sort_values(["Symbol", "Date"])


def load_corporate_actions(
    symbols: Iterable[str],
    dividend_fetcher: Optional[Callable[[str], pd.DataFrame]] = None,
    split_fetcher: Optional[Callable[[str], pd.DataFrame]] = None,
    insider_fetcher: Optional[Callable[[str], pd.DataFrame]] = None,
    thirteenf_fetcher: Optional[Callable[[str], pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Return merged corporate action datasets for ``symbols``."""

    frames = [
        load_dividends(symbols, fetcher=dividend_fetcher),
        load_splits(symbols, fetcher=split_fetcher),
        load_insider_filings(symbols, fetcher=insider_fetcher),
        load_13f_filings(symbols, fetcher=thirteenf_fetcher),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(
            columns=[
                "Date",
                "Symbol",
                "dividend",
                "split",
                "insider_trades",
                "institutional_holdings",
            ]
        )
    out = frames[0]
    for extra in frames[1:]:
        out = out.merge(extra, on=["Date", "Symbol"], how="outer")
    return out.sort_values(["Symbol", "Date"])


__all__ = [
    "load_dividends",
    "load_splits",
    "load_insider_filings",
    "load_13f_filings",
    "load_corporate_actions",
]
