"""Lightweight corporate actions data loaders.

This module ingests dividend payments, stock split ratios, insider
transaction summaries and 13F institutional holdings from either local CSV
files or public web APIs.  The functions are intentionally simple so they
work in minimal test environments.  Dataframes returned by the loaders are
aligned on ``Date`` and ``Symbol`` so they can be merged using
:func:`pandas.merge_asof`.

The default API implementation relies on the `Alpha Vantage
<https://www.alphavantage.co/>`_ endpoints which provide free access to
basic corporate action information using the ``demo`` key.  Callers may
provide custom ``fetcher`` callables to override the network behaviour or
supply pre‑fetched data during tests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import pandas as pd

try:  # pragma: no cover - requests optional
    import requests
except Exception:  # pragma: no cover - requests optional
    requests = None  # type: ignore

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


# ---------------------------------------------------------------------------
# API fetchers

def _fetch_dividend_api(symbol: str) -> pd.DataFrame:
    """Fetch dividend history from Alpha Vantage."""

    if requests is None:  # pragma: no cover - requests missing
        return pd.DataFrame()
    url = "https://www.alphavantage.co/query"
    params = {"function": "DIVIDENDS", "symbol": symbol, "apikey": "demo"}
    try:
        data = requests.get(url, params=params, timeout=10).json().get("data", [])
    except Exception:  # pragma: no cover - network errors
        logger.warning("Failed to fetch dividends for %s", symbol)
        return pd.DataFrame()
    if not isinstance(data, list):
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    date_col = "ex_dividend_date" if "ex_dividend_date" in df.columns else df.columns[0]
    amt_col = "amount" if "amount" in df.columns else df.columns[-1]
    df["Date"] = pd.to_datetime(df[date_col], utc=True)
    df = df[["Date", amt_col]].rename(columns={amt_col: "dividend"})
    return df


def _fetch_split_api(symbol: str) -> pd.DataFrame:
    """Fetch split history using daily adjusted time series from Alpha Vantage."""

    if requests is None:  # pragma: no cover - requests missing
        return pd.DataFrame()
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "apikey": "demo",
        "outputsize": "compact",
    }
    try:
        data = requests.get(url, params=params, timeout=10).json().get(
            "Time Series (Daily)",
            {},
        )
    except Exception:  # pragma: no cover - network errors
        logger.warning("Failed to fetch splits for %s", symbol)
        return pd.DataFrame()
    rows: List[dict] = []
    for date, vals in data.items():
        coeff = float(vals.get("8. split coefficient", 1.0))
        if coeff != 1.0:
            rows.append({"Date": pd.to_datetime(date, utc=True), "split": coeff})
    return pd.DataFrame(rows)


def _fetch_insider_api(symbol: str) -> pd.DataFrame:
    """Fetch insider sentiment data from Alpha Vantage."""

    if requests is None:  # pragma: no cover - requests missing
        return pd.DataFrame()
    url = "https://www.alphavantage.co/query"
    params = {"function": "INSIDER_SENTIMENT", "symbol": symbol, "apikey": "demo"}
    try:
        data = requests.get(url, params=params, timeout=10).json().get("data", [])
    except Exception:  # pragma: no cover - network errors
        logger.warning("Failed to fetch insider data for %s", symbol)
        return pd.DataFrame()
    if not isinstance(data, list):
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    if {"year", "month", "change"}.issubset(df.columns):
        df["Date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str) + "-01",
            utc=True,
        )
        df = df[["Date", "change"]].rename(columns={"change": "insider_trades"})
        return df
    # Fallback: attempt generic columns
    for col in df.columns:
        if "date" in col.lower():
            df["Date"] = pd.to_datetime(df[col], utc=True)
        if "change" in col.lower() or "shares" in col.lower():
            df["insider_trades"] = pd.to_numeric(df[col], errors="coerce")
    return df[["Date", "insider_trades"]].dropna()


def _fetch_13f_api(symbol: str) -> pd.DataFrame:
    """Fetch institutional holdings (13F) from Alpha Vantage."""

    if requests is None:  # pragma: no cover - requests missing
        return pd.DataFrame()
    url = "https://www.alphavantage.co/query"
    params = {"function": "INSTITUTIONAL_HOLDINGS", "symbol": symbol, "apikey": "demo"}
    try:
        data = requests.get(url, params=params, timeout=10).json().get("data", [])
    except Exception:  # pragma: no cover - network errors
        logger.warning("Failed to fetch 13F data for %s", symbol)
        return pd.DataFrame()
    if not isinstance(data, list):
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    val_col = next((c for c in df.columns if "shares" in c.lower()), None)
    if date_col and val_col:
        df["Date"] = pd.to_datetime(df[date_col], utc=True)
        df = df[["Date", val_col]].rename(columns={val_col: "institutional_holdings"})
        return df
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Public loader functions

def load_dividends(
    symbols: Iterable[str], fetcher: Optional[Callable[[str], pd.DataFrame]] = None
) -> pd.DataFrame:
    """Return dividend data for ``symbols`` as ``Date``/``Symbol``/``dividend``."""

    if fetcher is None:
        fetcher = _fetch_dividend_api
    frames: List[pd.DataFrame] = []
    for sym in symbols:
        df = _read_category_csv(sym, "dividends")
        if df.empty:
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

    if fetcher is None:
        fetcher = _fetch_split_api
    frames: List[pd.DataFrame] = []
    for sym in symbols:
        df = _read_category_csv(sym, "splits")
        if df.empty:
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

    if fetcher is None:
        fetcher = _fetch_insider_api
    frames: List[pd.DataFrame] = []
    for sym in symbols:
        df = _read_category_csv(sym, "insider")
        if df.empty:
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

    if fetcher is None:
        fetcher = _fetch_13f_api
    frames: List[pd.DataFrame] = []
    for sym in symbols:
        df = _read_category_csv(sym, "thirteenf")
        if df.empty:
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
