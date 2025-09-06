"""Volatility skew feature loader."""

from __future__ import annotations

import logging

import pandas as pd

from .options_vol import load_options_data

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

REQUIREMENTS = ResourceCapabilities(cpus=16, memory_gb=64.0, has_gpu=False, gpu_count=0)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Merge volatility skew data into ``df``."""

    if "vol_skew" in df.columns:
        df["vol_skew"] = df["vol_skew"].ffill().fillna(0.0)
        return df

    if "Symbol" not in df.columns:
        df["vol_skew"] = 0.0
        return df

    options = load_options_data(sorted(df["Symbol"].unique()))
    if options.empty or "vol_skew" not in options.columns:
        df["vol_skew"] = 0.0
        return df

    options = options[["Date", "Symbol", "vol_skew"]].rename(columns={"Date": "skew_date"})
    df = pd.merge_asof(
        df.sort_values("Timestamp"),
        options.sort_values("skew_date"),
        left_on="Timestamp",
        right_on="skew_date",
        by="Symbol",
        direction="backward",
    ).drop(columns=["skew_date"])

    if "vol_skew" in df.columns:
        df["vol_skew"] = df["vol_skew"].ffill().fillna(0.0)
    else:
        df["vol_skew"] = 0.0
    return df


__all__ = ["compute", "REQUIREMENTS"]
