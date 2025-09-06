"""Utilities for loading options implied volatility and skew."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from utils.resource_monitor import ResourceCapabilities
except Exception:  # pragma: no cover - optional during certain tests
    class ResourceCapabilities:  # type: ignore
        def __init__(self, cpus: int = 0, memory_gb: float = 0.0, has_gpu: bool = False, gpu_count: int = 0) -> None:
            self.cpus = cpus
            self.memory_gb = memory_gb
            self.has_gpu = has_gpu
            self.gpu_count = gpu_count

logger = logging.getLogger(__name__)


# heavy options datasets are only available on large instances
REQUIREMENTS = ResourceCapabilities(cpus=16, memory_gb=64.0, has_gpu=False, gpu_count=0)


def _read_local_csv(symbol: str) -> pd.DataFrame:
    """Load options data for ``symbol`` from local CSVs.

    The loader searches a couple of conventional locations used in tests
    and deployments.  Failure to read a file simply results in an empty
    dataframe which is handled by the caller.
    """

    paths = [
        Path("dataset") / "options" / f"{symbol}.csv",
        Path("data") / "options" / f"{symbol}.csv",
    ]
    for path in paths:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:  # pragma: no cover - read failure should not crash
                logger.warning("Failed to load options data for %s from %s", symbol, path)
                return pd.DataFrame()
    return pd.DataFrame()


def load_options_data(symbols: Iterable[str]) -> pd.DataFrame:
    """Load options implied volatility or skew for ``symbols``.

    The returned dataframe contains ``Date`` and ``Symbol`` columns plus
    any of ``implied_vol`` or ``vol_skew`` depending on availability of
    the underlying data.  Missing series are silently skipped.
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
        return pd.DataFrame(columns=["Date", "Symbol", "implied_vol", "vol_skew"])

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["Symbol", "Date"])


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Merge options implied volatility and skew into ``df``.

    For symbols where options data is available, the function performs a
    ``merge_asof`` on timestamp and forward fills missing values.  When no
    data is available the columns are filled with ``0`` so downstream
    models can rely on their presence.
    """

    if "Symbol" not in df.columns:
        df["implied_vol"] = 0.0
        df["vol_skew"] = 0.0
        return df

    options = load_options_data(sorted(df["Symbol"].unique()))
    if options.empty:
        df["implied_vol"] = 0.0
        df["vol_skew"] = 0.0
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

    for col in ["implied_vol", "vol_skew"]:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(0.0)
        else:
            df[col] = 0.0

    return df


__all__ = ["load_options_data", "compute", "REQUIREMENTS"]
