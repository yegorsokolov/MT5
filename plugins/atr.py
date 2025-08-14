"""Average True Range feature plugin.

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
def add_atr_stops(df):
    """Add ATR trailing stop columns."""
    cfg = load_config()
    if not cfg.get("use_atr", True):
        return df

    if "mid" not in df.columns:
        if {"Bid", "Ask"}.issubset(df.columns):
            df["mid"] = (df["Bid"] + df["Ask"]) / 2
        else:
            return df

    if {"atr_14", "atr_stop_long", "atr_stop_short"}.issubset(df.columns):
        return df

    period = cfg.get("atr_period", 14)
    mult = cfg.get("atr_mult", 3)
    tr = df["mid"].diff().abs()
    df["atr_14"] = tr.rolling(period).mean()
    df["atr_stop_long"] = df["mid"] - df["atr_14"] * mult
    df["atr_stop_short"] = df["mid"] + df["atr_14"] * mult
    return df
