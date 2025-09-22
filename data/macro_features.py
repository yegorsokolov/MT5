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
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Union

import pandas as pd

try:  # pragma: no cover - optional dependency at runtime
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore

from .macro_sources import SeriesSpec, fetch_series_data, parse_series_spec

DEFAULT_DATA_DELAY = pd.Timedelta(hours=1)

logger = logging.getLogger(__name__)


def _sanitise_symbol(symbol: str) -> str:
    """Return a filesystem friendly version of ``symbol``."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", symbol)


def _read_local_csv(symbol: str, alias: Optional[str] = None) -> pd.DataFrame:
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

    names = [symbol]
    if alias and alias not in names:
        names.append(alias)
    sanitised = _sanitise_symbol(symbol)
    if sanitised not in names:
        names.append(sanitised)
    if alias:
        sanitised_alias = _sanitise_symbol(alias)
        if sanitised_alias not in names:
            names.append(sanitised_alias)

    bases = [
        Path("data"),
        Path("dataset"),
        Path("data") / "macro",
        Path("dataset") / "macro",
    ]
    seen = set()
    for base in bases:
        for name in names:
            path = base / f"{name}.csv"
            if path in seen:
                continue
            seen.add(path)
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


def _cache_remote_series(spec: SeriesSpec, df: pd.DataFrame) -> None:
    if df.empty:
        return
    out_dir = Path("data") / "macro"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_name = _sanitise_symbol(spec.raw)
    try:  # pragma: no cover - filesystem race conditions
        df.to_csv(out_dir / f"{cache_name}.csv", index=False)
    except Exception:
        logger.debug("Failed to cache macro series %s", spec.raw, exc_info=True)


DelayValue = Union[pd.Timedelta, str, int, float, None]


def _normalise_delay(value: DelayValue) -> pd.Timedelta:
    """Return a non-negative :class:`~pandas.Timedelta` from ``value``."""

    if value is None:
        return pd.Timedelta(0)
    if isinstance(value, pd.Timedelta):
        return value if value >= pd.Timedelta(0) else pd.Timedelta(0)
    if isinstance(value, (int, float)):
        if value <= 0:
            return pd.Timedelta(0)
        return pd.to_timedelta(value, unit="s")
    try:
        delay = pd.Timedelta(value)
    except (TypeError, ValueError):
        logger.warning("Invalid macro data delay %s; defaulting to 0", value)
        return pd.Timedelta(0)
    if delay < pd.Timedelta(0):
        logger.warning("Negative macro data delay %s; defaulting to 0", value)
        return pd.Timedelta(0)
    return delay


def _normalise_delay_config(
    value: Union[DelayValue, Mapping[str, DelayValue]]
) -> tuple[pd.Timedelta, dict[str, pd.Timedelta]]:
    """Return a base delay and per-column overrides."""

    if isinstance(value, Mapping):
        default_delay = DEFAULT_DATA_DELAY
        overrides: dict[str, pd.Timedelta] = {}
        for key, delay_value in value.items():
            normalised = _normalise_delay(delay_value)
            key_str = str(key)
            if key_str.lower() in {"*", "default", "__default__"}:
                default_delay = normalised
            else:
                overrides[key_str] = normalised
        return default_delay, overrides

    return _normalise_delay(value), {}


def _delay_for_column(
    column: str, overrides: Mapping[str, pd.Timedelta], default_delay: pd.Timedelta
) -> pd.Timedelta:
    """Resolve the effective delay for ``column``."""

    if column in overrides:
        return overrides[column]
    prefixed = f"macro_{column}"
    if prefixed in overrides:
        return overrides[prefixed]
    if column.startswith("macro_"):
        bare = column[len("macro_") :]
        if bare in overrides:
            return overrides[bare]
    return default_delay


def _date_to_str(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, pd.Timestamp):
        dt = value.to_pydatetime()
    elif isinstance(value, datetime):
        dt = value
    else:
        return str(value)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d")


def load_macro_series(
    symbols: List[str],
    *,
    session: "httpx.Client | None" = None,
    start: Optional[object] = None,
    end: Optional[object] = None,
) -> pd.DataFrame:
    """Load macroeconomic time series for the given symbols.

    The loader first searches for locally cached CSV files.  When a symbol is
    prefixed by ``provider::`` the relevant API integration from
    :mod:`data.macro_sources` is used.  Successful remote fetches are cached for
    subsequent offline runs.

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

    specs = [parse_series_spec(sym) for sym in symbols]
    frames: List[pd.DataFrame] = []
    start_str = _date_to_str(start)
    end_str = _date_to_str(end)

    client = session
    created_client = False
    if client is None and any(spec.provider not in {"", "local", "csv"} for spec in specs):
        if httpx is None:  # pragma: no cover - exercised when httpx missing
            logger.debug("httpx unavailable – remote macro series disabled")
        else:
            client = httpx.Client(timeout=20.0)
            created_client = True

    for spec in specs:
        df = _read_local_csv(spec.raw, spec.column_name())
        if df.empty and spec.provider not in {"", "local", "csv"}:
            if client is None:
                logger.debug("Skipping remote fetch for %s – no HTTP client", spec.raw)
                continue
            remote = fetch_series_data(spec, client, start=start_str, end=end_str)
            if remote.empty:
                continue
            _cache_remote_series(spec, remote)
            df = remote
        if df.empty:
            continue
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        value_cols = [c for c in df.columns if c != "Date"]
        if not value_cols:
            continue
        value_col = value_cols[0]
        col_name = spec.column_name()
        if value_col != col_name:
            df = df[["Date", value_col]].rename(columns={value_col: col_name})
        else:
            df = df[["Date", value_col]]
        frames.append(df)

    if created_client and client is not None:
        try:  # pragma: no cover - cleanup path only
            client.close()
        except Exception:
            logger.debug("Failed closing httpx client", exc_info=True)

    if not frames:
        expected_cols = ["Date"] + [spec.column_name() for spec in specs]
        return pd.DataFrame(columns=expected_cols)

    out = frames[0]
    for extra in frames[1:]:
        out = out.merge(extra, on="Date", how="outer")
    return out.sort_values("Date")


def load_macro_features(
    df: pd.DataFrame,
    macro_df: Optional[pd.DataFrame] = None,
    data_delay: Union[DelayValue, Mapping[str, DelayValue]] = DEFAULT_DATA_DELAY,
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
    data_delay:
        Expected publication delay for the macro data.  Observations are only
        considered available once ``Timestamp`` exceeds their ``Date`` by at
        least this delay.  Numeric values are interpreted as seconds.  When a
        mapping is provided, delays can be configured per macro column with the
        special keys ``"*"``/``"default"`` overriding the base delay.  Defaults
        to one hour to reflect the latency of the newly integrated APIs.

    Returns
    -------
    pd.DataFrame
        ``df`` enriched with ``macro_`` prefixed columns.  Missing values are
        forward filled then remaining gaps are set to ``0`` so downstream
        models can rely on their presence.
    """

    base_delay, delay_overrides = _normalise_delay_config(data_delay)

    if macro_df is None:
        macro_df = _load_all_local()

    if macro_df.empty:
        # Ensure expected prefix columns exist if we can infer any names
        for col in [c for c in macro_df.columns if c not in {"Date", "Region"}]:
            df[f"macro_{col}"] = 0.0
        return df

    macro_df = macro_df.rename(columns={"Date": "macro_date"})
    sort_cols = ["macro_date"]
    if "Region" in macro_df.columns:
        sort_cols.insert(0, "Region")
    macro_df = macro_df.sort_values(sort_cols)

    df_sorted = df.sort_values("Timestamp").copy()
    region_key: Optional[str] = None
    if "Region" in df_sorted.columns and "Region" in macro_df.columns:
        region_key = "Region"

    value_cols = [c for c in macro_df.columns if c not in {"macro_date", "Region"}]
    result = df_sorted

    for col in value_cols:
        delay = _delay_for_column(col, delay_overrides, base_delay)
        macro_cols = ["macro_date", col]
        if region_key:
            macro_cols.append(region_key)
        col_macro = macro_df[macro_cols].dropna(subset=[col]).copy()
        if col_macro.empty:
            result[f"macro_{col}"] = 0.0
            continue
        available_col = "_macro_available_ts"
        if delay > pd.Timedelta(0):
            col_macro[available_col] = col_macro["macro_date"] + delay
        else:
            col_macro[available_col] = col_macro["macro_date"]
        sort_keys = [available_col]
        if region_key:
            sort_keys.insert(0, region_key)
        col_macro = col_macro.sort_values(sort_keys)

        merge_kwargs = {
            "left_on": "Timestamp",
            "right_on": available_col,
            "direction": "backward",
        }
        if region_key:
            merge_kwargs["by"] = region_key

        merged = pd.merge_asof(result, col_macro, **merge_kwargs)
        merged[f"macro_{col}"] = merged[col].ffill().fillna(0.0)
        result = merged.drop(columns=[col, "macro_date", available_col], errors="ignore")

    return result


__all__ = ["load_macro_series", "load_macro_features"]
