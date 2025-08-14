"""Donchian channel feature plugin.

min_cpus: 1
min_mem_gb: 0.1
requires_gpu: false
"""

MIN_CPUS = 1
MIN_MEM_GB = 0.1
REQUIRES_GPU = False

from . import register_feature
from utils import load_config

@register_feature
def add_donchian_channels(df):
    """Add Donchian channel features."""
    cfg = load_config()
    if not cfg.get("use_donchian", True):
        return df

    if "mid" not in df.columns:
        if {"Bid", "Ask"}.issubset(df.columns):
            df["mid"] = (df["Bid"] + df["Ask"]) / 2
        else:
            return df

    if {"donchian_high", "donchian_low", "donchian_break"}.issubset(df.columns):
        return df

    period = cfg.get("donchian_period", 20)
    df["donchian_high"] = df["mid"].rolling(period).max()
    df["donchian_low"] = df["mid"].rolling(period).min()
    dc_up = df["mid"] > df["donchian_high"].shift(1)
    dc_down = df["mid"] < df["donchian_low"].shift(1)
    df["donchian_break"] = 0
    df.loc[dc_up, "donchian_break"] = 1
    df.loc[dc_down, "donchian_break"] = -1
    return df
