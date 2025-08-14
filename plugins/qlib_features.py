"""Qlib technical factor feature plugin.

min_cpus: 2
min_mem_gb: 2
requires_gpu: false
"""

MIN_CPUS = 2
MIN_MEM_GB = 2.0
REQUIRES_GPU = False

from . import register_feature
from utils import load_config
import pandas as pd
import logging

logger = logging.getLogger(__name__)

@register_feature
def add_qlib_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical factors computed via Microsoft Qlib when enabled."""
    cfg = load_config()
    if not cfg.get("use_qlib_features", False):
        return df

    try:
        from qlib.contrib import ta  # type: ignore
    except Exception as e:  # pragma: no cover - heavy optional dependency
        logger.warning("Qlib not available: %s", e)
        return df

    if "mid" not in df.columns and {"Bid", "Ask"}.issubset(df.columns):
        df = df.assign(mid=(df["Bid"] + df["Ask"]) / 2)

    if "mid" not in df.columns:
        return df

    try:
        df["qlib_ma10"] = ta.MA(df["mid"], window=10)
        df["qlib_ma30"] = ta.MA(df["mid"], window=30)
        df["qlib_rsi14"] = ta.RSI(df["mid"], window=14)
    except Exception as e:  # pragma: no cover - runtime dependency
        logger.warning("Failed to compute qlib factors: %s", e)
    return df
