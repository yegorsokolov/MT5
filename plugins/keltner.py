"""Keltner channel feature plugin.

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

@register_feature
def add_keltner_channels(df: pd.DataFrame) -> pd.DataFrame:
    """Add Keltner Channel features when enabled."""
    cfg = load_config()
    if not cfg.get("use_keltner", False):
        return df

    if "mid" not in df.columns:
        if {"Bid", "Ask"}.issubset(df.columns):
            df["mid"] = (df["Bid"] + df["Ask"]) / 2
        else:
            return df

    if {"keltner_high", "keltner_low", "keltner_break"}.issubset(df.columns):
        return df

    period = cfg.get("keltner_period", 20)
    mult = cfg.get("keltner_mult", 2)

    ma = df["mid"].rolling(period).mean()
    atr = df["mid"].diff().abs().rolling(period).mean()
    df["keltner_high"] = ma + atr * mult
    df["keltner_low"] = ma - atr * mult

    up = df["mid"] > df["keltner_high"].shift(1)
    down = df["mid"] < df["keltner_low"].shift(1)
    df["keltner_break"] = 0
    df.loc[up, "keltner_break"] = 1
    df.loc[down, "keltner_break"] = -1
    return df
