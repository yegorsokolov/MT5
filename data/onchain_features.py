"""Blockchain on-chain metrics integration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


def _read_local_csv(symbol: str) -> pd.DataFrame:
    """Load on-chain metrics for ``symbol`` from local CSV files."""
    paths = [
        Path("dataset") / "onchain" / f"{symbol}.csv",
        Path("data") / "onchain" / f"{symbol}.csv",
    ]
    for path in paths:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:  # pragma: no cover - read failure
                logger.warning("Failed to load on-chain data for %s from %s", symbol, path)
                return pd.DataFrame()
    return pd.DataFrame()


def load_onchain_data(symbols: Iterable[str]) -> pd.DataFrame:
    """Load blockchain metrics for ``symbols``.

    CSV files should contain ``Date`` and an ``active_addresses`` column. Metrics
    are typically only available for crypto assets and the function silently
    skips symbols without data.
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
        return pd.DataFrame(columns=["Date", "Symbol", "active_addresses"])

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["Symbol", "Date"])


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Merge on-chain metrics into ``df``.

    Adds an ``active_addresses`` column aligned on ``Timestamp``. Missing values
    are forward-filled and remaining gaps filled with zeros.
    """

    if "Symbol" not in df.columns:
        df["active_addresses"] = 0.0
        return df

    onchain = load_onchain_data(sorted(df["Symbol"].unique()))
    if onchain.empty:
        df["active_addresses"] = 0.0
        return df

    onchain = onchain.rename(columns={"Date": "chain_date"})
    df = pd.merge_asof(
        df.sort_values("Timestamp"),
        onchain.sort_values("chain_date"),
        left_on="Timestamp",
        right_on="chain_date",
        by="Symbol",
        direction="backward",
    ).drop(columns=["chain_date"])

    if "active_addresses" in df.columns:
        df["active_addresses"] = df["active_addresses"].ffill().fillna(0.0)
    else:
        df["active_addresses"] = 0.0
    return df


__all__ = ["load_onchain_data", "compute"]
