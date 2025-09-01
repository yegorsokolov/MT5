"""Alternative dataset loader utilities.

This module fetches alternative data sources such as options implied
volatility, blockchain on-chain metrics and environmental, social and
governance (ESG) scores.  Data is loaded from local CSV files when
available and otherwise omitted.  Each loader returns a tidy dataframe
with a ``Date`` column and the value column for easy merging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

import pandas as pd

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
                logger.warning("Failed to read %s data for %s from %s", category, symbol, path)
                return pd.DataFrame()
    return pd.DataFrame()


def _load_generic(symbols: Iterable[str], category: str, value_col: str) -> pd.DataFrame:
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


def load_alt_data(symbols: Iterable[str]) -> pd.DataFrame:
    """Return merged alternative datasets for ``symbols``."""

    frames = [
        load_options_implied_vol(symbols),
        load_onchain_metrics(symbols),
        load_esg_scores(symbols),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(
            columns=["Date", "Symbol", "implied_vol", "active_addresses", "esg_score"]
        )

    out = frames[0]
    for extra in frames[1:]:
        out = out.merge(extra, on=["Date", "Symbol"], how="outer")
    return out.sort_values(["Symbol", "Date"])


__all__ = [
    "load_options_implied_vol",
    "load_onchain_metrics",
    "load_esg_scores",
    "load_alt_data",
]
