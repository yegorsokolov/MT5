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

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Set
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd
import requests

from core.context_hub import context_hub

logger = logging.getLogger(__name__)


_KALSHI_BASE_URL = "https://kalshi-public-docs.s3.amazonaws.com/reporting"
_KALSHI_CACHE_DIR = Path("data") / "kalshi"
_KALSHI_MAPPING_FILES = [
    Path("data") / "kalshi_mapping.json",
    Path("dataset") / "kalshi_mapping.json",
]


def _load_cached_json(path: Path) -> list[dict[str, object]]:
    try:
        raw = json.loads(path.read_text())
        if isinstance(raw, list):
            return raw
    except Exception:  # pragma: no cover - cache corruption is non fatal
        logger.warning("Failed to read cached Kalshi data from %s", path)
    return []


def _download_kalshi_day(day: date) -> list[dict[str, object]]:
    cache_path = _KALSHI_CACHE_DIR / f"market_data_{day.isoformat()}.json"
    if cache_path.exists():
        cached = _load_cached_json(cache_path)
        if cached:
            return cached

    url = f"{_KALSHI_BASE_URL}/market_data_{day.isoformat()}.json"
    try:
        with urlopen(url) as resp:  # type: ignore[arg-type]
            data = json.load(resp)
    except (HTTPError, URLError):  # pragma: no cover - network best effort
        return []
    except Exception:  # pragma: no cover - JSON parse error
        logger.warning("Unable to decode Kalshi data for %s", day)
        return []

    if not isinstance(data, list):
        return []

    try:  # pragma: no cover - cache directory may not be writable
        _KALSHI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data))
    except Exception:
        logger.debug("Failed to cache Kalshi data for %s", day)
    return data


def _kalshi_frame(
    day: date,
    *,
    fetcher: Optional[Callable[[date], Sequence[Mapping[str, object]]]] = None,
) -> pd.DataFrame:
    raw: Sequence[Mapping[str, object]] | list[dict[str, object]]
    if fetcher is not None:
        try:
            raw = fetcher(day)
        except Exception:  # pragma: no cover - custom fetcher failure
            logger.warning("Kalshi fetcher failed for %s", day)
            raw = []
    else:
        raw = _download_kalshi_day(day)

    if not raw:
        return pd.DataFrame()

    frame = pd.DataFrame(list(raw))
    required = {"date", "report_ticker"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()

    frame["Date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["report_ticker"] = frame["report_ticker"].astype(str).str.upper()

    for col in ["open_interest", "daily_volume", "block_volume"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
        else:
            frame[col] = 0.0
    for col in ["high", "low"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            frame[col] = pd.NA

    frame = frame.dropna(subset=["Date", "report_ticker"])
    return frame[
        [
            "Date",
            "report_ticker",
            "open_interest",
            "daily_volume",
            "block_volume",
            "high",
            "low",
        ]
    ].sort_values(["report_ticker", "Date"])


def _symbol_variants(symbol: str) -> Set[str]:
    sym = symbol.upper()
    variants: Set[str] = {sym, sym.replace("/", ""), sym.replace("-", "")}
    for suffix in ("USD", "USDT", "USDC", "PERP"):
        if sym.endswith(suffix) and len(sym) > len(suffix):
            variants.add(sym[: -len(suffix)])
    return {v for v in variants if v}


def _guess_kalshi_mapping(symbols: Sequence[str], tickers: Set[str]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for sym in symbols:
        candidates: Set[str] = set()
        variants = _symbol_variants(sym)
        for ticker in tickers:
            for variant in variants:
                if ticker == variant or ticker.startswith(variant) or variant.startswith(ticker):
                    candidates.add(ticker)
        if candidates:
            mapping[sym] = sorted(candidates)
    return mapping


def _load_kalshi_mapping(
    symbols: Sequence[str], tickers: Set[str]
) -> dict[str, list[str]]:
    for path in _KALSHI_MAPPING_FILES:
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text())
        except Exception:  # pragma: no cover - config parsing failure
            logger.warning("Failed to parse Kalshi mapping file %s", path)
            continue
        mapping: dict[str, list[str]] = {}
        for sym, values in raw.items():
            if not values:
                continue
            if isinstance(values, str):
                values = [values]
            mapping[sym] = [str(v).upper() for v in values]
        if mapping:
            return {sym: ticks for sym, ticks in mapping.items() if ticks}
    return _guess_kalshi_mapping(symbols, tickers)


def _empty_kalshi_frame() -> pd.DataFrame:
    columns = [
        "Date",
        "Symbol",
        "kalshi_total_open_interest",
        "kalshi_total_daily_volume",
        "kalshi_total_block_volume",
        "kalshi_total_high",
        "kalshi_total_low",
        "kalshi_market_count",
        "kalshi_open_interest",
        "kalshi_daily_volume",
        "kalshi_block_volume",
        "kalshi_high",
        "kalshi_low",
    ]
    return pd.DataFrame(columns=columns)


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


def load_kalshi_markets(
    symbols: Iterable[str],
    lookback_days: int = 90,
    *,
    fetcher: Optional[Callable[[date], Sequence[Mapping[str, object]]]] = None,
    mapping: Optional[Mapping[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """Load aggregated Kalshi prediction market metrics for ``symbols``.

    The loader retrieves public daily market data snapshots published by
    Kalshi.  Each snapshot provides contract open interest, trading volume and
    the high/low price range for every listed market on that day.  The data is
    aggregated per ``report_ticker`` (Kalshi's market group identifier) and
    then mapped onto the requested trading symbols using either an explicit
    mapping file (``data/kalshi_mapping.json``) or a heuristic that matches
    overlapping ticker prefixes (``BTC`` → ``BTCUSD`` etc.).

    Parameters
    ----------
    symbols:
        Iterable of trading symbols.  The loader derives Kalshi tickers that
        should influence each symbol.
    lookback_days:
        Number of calendar days to download.  Defaults to 90 which keeps
        network usage modest while still providing context for feature merges.
    fetcher:
        Optional callback returning raw JSON-like objects for a given
        :class:`datetime.date`.  When omitted the public S3 snapshots are used
        and cached under ``data/kalshi`` for offline reuse.
    mapping:
        Optional explicit mapping of symbol -> Kalshi ``report_ticker`` values.

    Returns
    -------
    pd.DataFrame
        Dataframe sorted by ``Symbol`` and ``Date`` containing both global
        totals (columns prefixed with ``kalshi_total_``) and symbol specific
        aggregates.  Missing values are filled with ``0`` so downstream merges
        can rely on their presence.
    """

    symbol_list = sorted({sym for sym in symbols if sym})
    if not symbol_list:
        context_hub.update(
            "kalshi",
            {
                "latest_date": None,
                "kalshi_total_open_interest": 0.0,
                "kalshi_total_daily_volume": 0.0,
                "kalshi_total_block_volume": 0.0,
                "kalshi_total_high": 0.0,
                "kalshi_total_low": 0.0,
                "kalshi_market_count": 0,
            },
        )
        return _empty_kalshi_frame()

    lookback_days = max(int(lookback_days), 1)
    end = pd.Timestamp.utcnow().date()
    start = end - timedelta(days=lookback_days - 1)

    frames: list[pd.DataFrame] = []
    for day in pd.date_range(start, end, freq="D"):
        frame = _kalshi_frame(day.date(), fetcher=fetcher)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        context_hub.update(
            "kalshi",
            {
                "latest_date": None,
                "kalshi_total_open_interest": 0.0,
                "kalshi_total_daily_volume": 0.0,
                "kalshi_total_block_volume": 0.0,
                "kalshi_total_high": 0.0,
                "kalshi_total_low": 0.0,
                "kalshi_market_count": 0,
            },
        )
        return _empty_kalshi_frame()

    raw = pd.concat(frames, ignore_index=True)
    if raw.empty:
        return _empty_kalshi_frame()

    tickers = set(raw["report_ticker"].dropna().astype(str))
    if mapping is None:
        mapping = _load_kalshi_mapping(symbol_list, tickers)
    else:
        mapping = {
            sym: [str(t).upper() for t in values]
            for sym, values in mapping.items()
            if values
        }

    grouped = raw.groupby(["Date", "report_ticker"], as_index=False).agg(
        open_interest=("open_interest", "sum"),
        daily_volume=("daily_volume", "sum"),
        block_volume=("block_volume", "sum"),
        high=("high", "max"),
        low=("low", "min"),
    )

    totals = grouped.groupby("Date", as_index=False).agg(
        kalshi_total_open_interest=("open_interest", "sum"),
        kalshi_total_daily_volume=("daily_volume", "sum"),
        kalshi_total_block_volume=("block_volume", "sum"),
        kalshi_total_high=("high", "max"),
        kalshi_total_low=("low", "min"),
    )
    totals = totals.merge(
        grouped.groupby("Date")["report_ticker"].nunique().reset_index(name="kalshi_market_count"),
        on="Date",
        how="left",
    )
    totals[[
        "kalshi_total_open_interest",
        "kalshi_total_daily_volume",
        "kalshi_total_block_volume",
    ]] = totals[[
        "kalshi_total_open_interest",
        "kalshi_total_daily_volume",
        "kalshi_total_block_volume",
    ]].fillna(0.0)
    totals["kalshi_total_high"] = totals["kalshi_total_high"].fillna(0.0)
    totals["kalshi_total_low"] = totals["kalshi_total_low"].fillna(0.0)
    totals["kalshi_market_count"] = totals["kalshi_market_count"].fillna(0).astype(int)

    total_frames: list[pd.DataFrame] = []
    for sym in symbol_list:
        df_sym = totals.copy()
        df_sym["Symbol"] = sym
        total_frames.append(df_sym)
    totals_with_symbol = pd.concat(total_frames, ignore_index=True)

    symbol_frames: list[pd.DataFrame] = []
    for sym in symbol_list:
        tickers_for_sym = mapping.get(sym, []) if mapping else []
        if not tickers_for_sym:
            continue
        subset = grouped[grouped["report_ticker"].isin(tickers_for_sym)]
        if subset.empty:
            continue
        agg = subset.groupby("Date", as_index=False).agg(
            kalshi_open_interest=("open_interest", "sum"),
            kalshi_daily_volume=("daily_volume", "sum"),
            kalshi_block_volume=("block_volume", "sum"),
            kalshi_high=("high", "max"),
            kalshi_low=("low", "min"),
        )
        agg["Symbol"] = sym
        symbol_frames.append(agg)

    if symbol_frames:
        per_symbol = pd.concat(symbol_frames, ignore_index=True)
    else:
        per_symbol = pd.DataFrame(
            columns=[
                "Date",
                "kalshi_open_interest",
                "kalshi_daily_volume",
                "kalshi_block_volume",
                "kalshi_high",
                "kalshi_low",
                "Symbol",
            ]
        )

    result = totals_with_symbol.merge(per_symbol, on=["Date", "Symbol"], how="left")
    for col in [
        "kalshi_open_interest",
        "kalshi_daily_volume",
        "kalshi_block_volume",
        "kalshi_high",
        "kalshi_low",
    ]:
        result[col] = result[col].fillna(0.0)

    result = result.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    latest_totals = totals.sort_values("Date")
    if not latest_totals.empty:
        latest = latest_totals.iloc[-1]
        try:
            latest_date = pd.Timestamp(latest["Date"]).date().isoformat()
        except Exception:  # pragma: no cover - defensive conversion
            latest_date = str(latest["Date"])
        context_hub.update(
            "kalshi",
            {
                "latest_date": latest_date,
                "kalshi_total_open_interest": float(latest["kalshi_total_open_interest"]),
                "kalshi_total_daily_volume": float(latest["kalshi_total_daily_volume"]),
                "kalshi_total_block_volume": float(latest["kalshi_total_block_volume"]),
                "kalshi_total_high": float(latest["kalshi_total_high"]),
                "kalshi_total_low": float(latest["kalshi_total_low"]),
                "kalshi_market_count": int(latest["kalshi_market_count"]),
            },
        )
    else:
        context_hub.update(
            "kalshi",
            {
                "latest_date": None,
                "kalshi_total_open_interest": 0.0,
                "kalshi_total_daily_volume": 0.0,
                "kalshi_total_block_volume": 0.0,
                "kalshi_total_high": 0.0,
                "kalshi_total_low": 0.0,
                "kalshi_market_count": 0,
            },
        )

    return result


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
    ``retail_sales``, ``temperature``, macroeconomic indicators,
    ``news_sentiment`` plus Kalshi derived features such as
    ``kalshi_total_open_interest`` and ``kalshi_open_interest``.
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
        load_kalshi_markets(symbols),
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
                "kalshi_total_open_interest",
                "kalshi_total_daily_volume",
                "kalshi_total_block_volume",
                "kalshi_total_high",
                "kalshi_total_low",
                "kalshi_market_count",
                "kalshi_open_interest",
                "kalshi_daily_volume",
                "kalshi_block_volume",
                "kalshi_high",
                "kalshi_low",
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
    "load_kalshi_markets",
    "load_weather_series",
    "load_alt_data",
]
