"""Options market derived features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


def _read_local_csv(symbol: str) -> pd.DataFrame:
    """Load options implied volatility for ``symbol`` from local CSVs."""
    paths = [
        Path("dataset") / "options" / f"{symbol}.csv",
        Path("data") / "options" / f"{symbol}.csv",
    ]
    for path in paths:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:  # pragma: no cover - read failure
                logger.warning("Failed to load options data for %s from %s", symbol, path)
                return pd.DataFrame()
    return pd.DataFrame()


def load_options_data(symbols: Iterable[str]) -> pd.DataFrame:
    """Load options implied volatility for ``symbols``.

    The function searches for local CSV files containing ``Date`` and
    ``implied_vol`` columns. Missing series are skipped.
    """

    frames: list[pd.DataFrame] = []
    for sym in symbols:
        df = _read_local_csv(sym)
        if df.empty:
            continue
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df["Symbol"] = sym
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["Date", "Symbol", "implied_vol"])

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["Symbol", "Date"])


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Merge options implied volatility into ``df``.

    Adds an ``implied_vol`` column representing the 30-day options implied
    volatility. Missing data is forward-filled and remaining gaps set to zero.
    """

    if "Symbol" not in df.columns:
        df["implied_vol"] = 0.0
        return df

    options = load_options_data(sorted(df["Symbol"].unique()))
    if options.empty:
        df["implied_vol"] = 0.0
        return df

    options = options.rename(columns={"Date": "opt_date"})
    df = pd.merge_asof(
        df.sort_values("Timestamp"),
        options.sort_values("opt_date"),
        left_on="Timestamp",
        right_on="opt_date",
        by="Symbol",
        direction="backward",
    ).drop(columns=["opt_date"])

    if "implied_vol" in df.columns:
        df["implied_vol"] = df["implied_vol"].ffill().fillna(0.0)
    else:
        df["implied_vol"] = 0.0
    return df


__all__ = ["load_options_data", "compute"]
