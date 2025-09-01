"""Utilities for loading fundamental and macro data.

This module provides lightweight helpers to ingest financial statement
items, valuation ratios and macroeconomic series from either local CSV
files or public data APIs.  The functions are intentionally minimal and
avoid heavy dependencies so they can operate in constrained test
environments.  Each loader returns a tidy dataframe with a ``Date`` column
suitable for merging via :func:`pandas.merge_asof`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

import pandas as pd

logger = logging.getLogger(__name__)


def _read_category_csv(symbol: str, category: str) -> pd.DataFrame:
    """Read a CSV for ``symbol`` under ``dataset/<category>`` or ``data/<category>``.

    Parameters
    ----------
    symbol:
        Identifier of the series to load.
    category:
        Sub-directory name such as ``"fundamentals"`` or ``"ratios"``.

    Returns
    -------
    pd.DataFrame
        Dataframe loaded from the first existing path or an empty dataframe if
        no file is available.
    """

    paths = [
        Path("dataset") / category / f"{symbol}.csv",
        Path("data") / category / f"{symbol}.csv",
    ]
    for path in paths:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:  # pragma: no cover - local read failure
                logger.warning("Failed to read %s data for %s from %s", category, symbol, path)
                return pd.DataFrame()
    return pd.DataFrame()


def load_financial_statements(symbols: Iterable[str]) -> pd.DataFrame:
    """Load basic financial statement items for ``symbols``.

    The loader searches for CSV files containing columns such as ``revenue`` or
    ``net_income``.  Missing symbols are skipped silently.
    """

    frames: List[pd.DataFrame] = []
    for sym in symbols:
        df = _read_category_csv(sym, "fundamentals")
        if df.empty:
            continue
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df["Symbol"] = sym
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["Date", "Symbol"])

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["Symbol", "Date"])


def load_ratios(symbols: Iterable[str]) -> pd.DataFrame:
    """Load valuation ratios such as P/E and dividend yield."""

    frames: List[pd.DataFrame] = []
    for sym in symbols:
        df = _read_category_csv(sym, "ratios")
        if df.empty:
            continue
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df["Symbol"] = sym
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["Date", "Symbol"])

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["Symbol", "Date"])


def load_macro_series(symbols: List[str]) -> pd.DataFrame:
    """Wrapper around :func:`data.macro_features.load_macro_series`."""

    from .macro_features import load_macro_series as _load_macro

    return _load_macro(list(symbols))


def load_fundamental_data(symbols: Iterable[str]) -> pd.DataFrame:
    """Return combined fundamental data for ``symbols``.

    The returned dataframe contains the financial statement items, valuation
    ratios and a small set of macro series (GDP, CPI and interest rates).  All
    series are merged on the ``Date`` and ``Symbol`` columns.
    """

    statements = load_financial_statements(symbols)
    ratios = load_ratios(symbols)
    if statements.empty and ratios.empty:
        fund = pd.DataFrame(columns=["Date", "Symbol"])
    elif statements.empty:
        fund = ratios
    elif ratios.empty:
        fund = statements
    else:
        fund = pd.merge(statements, ratios, on=["Date", "Symbol"], how="outer")

    # Load a minimal macro set
    macro = load_macro_series(["gdp", "cpi", "interest_rate"])
    if not macro.empty:
        fund = fund.merge(macro, on="Date", how="left")
    else:
        for col in ["gdp", "cpi", "interest_rate"]:
            fund[col] = 0.0

    return fund.sort_values(["Symbol", "Date"]) if "Symbol" in fund.columns else fund


__all__ = [
    "load_financial_statements",
    "load_ratios",
    "load_macro_series",
    "load_fundamental_data",
]
