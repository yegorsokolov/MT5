"""Pair trading signal feature plugin.

min_cpus: 1
min_mem_gb: 0.1
requires_gpu: false
"""

MIN_CPUS = 1
MIN_MEM_GB = 0.1
REQUIRES_GPU = False

from . import register_feature
from utils import load_config
import pandas as pd
import numpy as np


@register_feature
def add_pair_trading_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate pair trading entry signals based on z-score thresholds."""
    cfg = load_config()
    if not cfg.get("use_pair_trading", False):
        df["pair_long"] = 0
        df["pair_short"] = 0
        return df

    long_th = cfg.get("pair_long_threshold", -2.0)
    short_th = cfg.get("pair_short_threshold", 2.0)
    df = df.sort_values("Timestamp").reset_index(drop=True)

    z_cols = [c for c in df.columns if c.startswith("pair_z_")]
    if not z_cols:
        df["pair_long"] = 0
        df["pair_short"] = 0
        return df

    long_sig = pd.Series(False, index=df.index)
    short_sig = pd.Series(False, index=df.index)
    time_index = df["Timestamp"]
    for col in z_cols:
        z_time = df.groupby("Timestamp")[col].first()
        long_cross = (z_time < long_th) & (z_time.shift(1) >= long_th)
        short_cross = (z_time > short_th) & (z_time.shift(1) <= short_th)
        long_series = long_cross.reindex(time_index).fillna(False)
        short_series = short_cross.reindex(time_index).fillna(False)
        long_sig |= long_series.values
        short_sig |= short_series.values

    df["pair_long"] = long_sig.astype(int)
    df["pair_short"] = short_sig.astype(int)
    return df

