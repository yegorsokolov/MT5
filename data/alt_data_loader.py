"""Alternative dataset loader utilities.

This module fetches a variety of alternative data sources used to enrich
the feature matrix.  Datasets include:

* **Options / On-chain / ESG** – legacy features already consumed by the
  system.
* **Shipping metrics** – e.g. `Freightos Baltic Index` or `Baltic Dry
  Index` style series stored as CSV files.
* **Retail transactions** – aggregated sales figures from government or
  commercial reports available as CSV downloads.
* **Weather series** – hourly temperature readings fetched from public
  APIs such as `Open‑Meteo <https://open-meteo.com/>`_ when a local CSV
  is not present.

Each loader returns a tidy dataframe with a ``Date`` column and value
columns for easy merging.  All dates are converted to timezone aware UTC
timestamps so they can be aligned with price history.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def _read_category_csv(symbol: str, category: str) -> pd.DataFrame:
    """Read CSV for ``symbol`` from ``dataset/<category>`` or ``data/<category>``."""

    paths = [
        Path("dataset") / category / f"{symbol}.csv",
        Path("data") / category / f"{symbol}.csv",
    ]
    for path in paths:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:  # pragma: no cover - local read failure
                logger.warning(
                    "Failed to read %s data for %s from %s", category, symbol, path
                )
                return pd.DataFrame()
    return pd.DataFrame()


def _load_generic(
    symbols: Iterable[str], category: str, value_col: str
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for sym in symbols:
        df = _read_category_csv(sym, category)
        if df.empty:
            continue
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df[["Date", value_col]].rename(columns={value_col: value_col})
        df["Symbol"] = sym
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["Date", "Symbol", value_col])
    return pd.concat(frames, ignore_index=True).sort_values(["Symbol", "Date"])


def load_options_implied_vol(symbols: Iterable[str]) -> pd.DataFrame:
    return _load_generic(symbols, "options", "implied_vol")


def load_onchain_metrics(symbols: Iterable[str]) -> pd.DataFrame:
    return _load_generic(symbols, "onchain", "active_addresses")


def load_esg_scores(symbols: Iterable[str]) -> pd.DataFrame:
    return _load_generic(symbols, "esg", "esg_score")


def load_shipping_metrics(symbols: Iterable[str]) -> pd.DataFrame:
    """Load maritime shipping indicators for ``symbols`` from CSV."""

    return _load_generic(symbols, "shipping", "shipping_metric")


def load_retail_transactions(symbols: Iterable[str]) -> pd.DataFrame:
    """Load retail sales aggregates for ``symbols`` from CSV."""

    return _load_generic(symbols, "retail", "retail_sales")


def load_macro_indicators(symbols: Iterable[str]) -> pd.DataFrame:
    """Load macroeconomic indicators such as GDP and CPI from CSV."""

    frames: List[pd.DataFrame] = []
    for sym in symbols:
        df = _read_category_csv(sym, "macro")
        if df.empty:
            continue
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df["Symbol"] = sym
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["Date", "Symbol", "gdp", "cpi", "interest_rate"])
    return pd.concat(frames, ignore_index=True).sort_values(["Symbol", "Date"])


def _fetch_news_sentiment_api(symbol: str) -> pd.DataFrame:
    """Fetch recent news sentiment for ``symbol`` using the Stocktwits API."""

    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    messages = resp.json().get("messages", [])
    if not messages:
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime([m.get("created_at") for m in messages], utc=True),
            "news_sentiment": [
                (
                    1
                    if m.get("entities", {}).get("sentiment", {}).get("basic")
                    == "Bullish"
                    else (
                        -1
                        if m.get("entities", {}).get("sentiment", {}).get("basic")
                        == "Bearish"
                        else 0
                    )
                )
                for m in messages
            ],
        }
    )
    df["Symbol"] = symbol
    return df


def load_news_sentiment(
    symbols: Iterable[str], fetcher: Optional[Callable[[str], pd.DataFrame]] = None
) -> pd.DataFrame:
    """Load news sentiment scores for ``symbols`` from CSV or via ``fetcher``."""

    if fetcher is None:
        fetcher = _fetch_news_sentiment_api

    frames: List[pd.DataFrame] = []
    for sym in symbols:
        df = _read_category_csv(sym, "news")
        if df.empty:
            try:
                df = fetcher(sym)
            except Exception:  # pragma: no cover - network errors
                logger.warning("Failed to fetch news sentiment for %s", sym)
                df = pd.DataFrame()
        if df.empty:
            continue
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df[["Date", "news_sentiment"]]
        df["Symbol"] = sym
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["Date", "Symbol", "news_sentiment"])
    return pd.concat(frames, ignore_index=True).sort_values(["Symbol", "Date"])


def _fetch_weather_api(symbol: str) -> pd.DataFrame:
    """Fetch hourly temperature series from Open‑Meteo for ``symbol``.

    The symbol is not used directly by the API; callers should map the
    symbol to appropriate coordinates if necessary.  For demonstration
    purposes we query the equator/prime‑meridian intersection.
    """

    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=0&longitude=0&hourly=temperature_2m"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json().get("hourly", {})
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(data.get("time", []), utc=True),
            "temperature": data.get("temperature_2m", []),
        }
    )
    df["Symbol"] = symbol
    return df


def load_weather_series(
    symbols: Iterable[str], fetcher: Optional[Callable[[str], pd.DataFrame]] = None
) -> pd.DataFrame:
    """Load weather data for ``symbols`` from CSV or via ``fetcher``.

    Parameters
    ----------
    symbols:
        Iterable of asset tickers.
    fetcher:
        Optional callable used to retrieve data when a local CSV is not
        available.  By default :func:`_fetch_weather_api` is used which
        queries the Open‑Meteo API.
    """

    if fetcher is None:
        fetcher = _fetch_weather_api

    frames: List[pd.DataFrame] = []
    for sym in symbols:
        df = _read_category_csv(sym, "weather")
        if df.empty:
            try:
                df = fetcher(sym)
            except Exception:  # pragma: no cover - network errors
                logger.warning("Failed to fetch weather data for %s", sym)
                df = pd.DataFrame()
        if df.empty:
            continue
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df[["Date", "temperature"]]
        df["Symbol"] = sym
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["Date", "Symbol", "temperature"])
    return pd.concat(frames, ignore_index=True).sort_values(["Symbol", "Date"])


def load_alt_data(
    symbols: Iterable[str],
    weather_fetcher: Optional[Callable[[str], pd.DataFrame]] = None,
    news_fetcher: Optional[Callable[[str], pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Return merged alternative datasets for ``symbols``.

    The resulting dataframe may contain any combination of the following
    columns depending on data availability: ``implied_vol``,
    ``active_addresses``, ``esg_score``, ``shipping_metric``,
    ``retail_sales`` and ``temperature``.
    """

    frames = [
        load_options_implied_vol(symbols),
        load_onchain_metrics(symbols),
        load_esg_scores(symbols),
        load_shipping_metrics(symbols),
        load_retail_transactions(symbols),
        load_weather_series(symbols, fetcher=weather_fetcher),
        load_macro_indicators(symbols),
        load_news_sentiment(symbols, fetcher=news_fetcher),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(
            columns=[
                "Date",
                "Symbol",
                "implied_vol",
                "active_addresses",
                "esg_score",
                "shipping_metric",
                "retail_sales",
                "temperature",
                "gdp",
                "cpi",
                "interest_rate",
                "news_sentiment",
            ]
        )

    out = frames[0]
    for extra in frames[1:]:
        out = out.merge(extra, on=["Date", "Symbol"], how="outer")
    return out.sort_values(["Symbol", "Date"])


__all__ = [
    "load_options_implied_vol",
    "load_onchain_metrics",
    "load_esg_scores",
    "load_shipping_metrics",
    "load_retail_transactions",
    "load_macro_indicators",
    "load_news_sentiment",
    "load_weather_series",
    "load_alt_data",
]
