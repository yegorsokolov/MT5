"""tsfresh statistical feature plugin.

min_cpus: 2
min_mem_gb: 2
requires_gpu: false
"""

from __future__ import annotations

MIN_CPUS = 2
MIN_MEM_GB = 2.0
REQUIRES_GPU = False

import pandas as pd
import logging

from . import register_feature
from utils import load_config

logger = logging.getLogger(__name__)


@register_feature
def add_tsfresh_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute selected tsfresh features from the mid price series."""
    cfg = load_config()
    if not cfg.get("use_tsfresh", False):
        return df

    try:
        from tsfresh.feature_extraction import feature_calculators as fc
    except Exception as e:  # pragma: no cover - optional dependency
        logger.warning("tsfresh not available: %s", e)
        return df

    if "mid" not in df.columns and {"Bid", "Ask"}.issubset(df.columns):
        df = df.assign(mid=(df["Bid"] + df["Ask"]) / 2)

    if "mid" not in df.columns:
        return df

    window = cfg.get("tsfresh_window", 30)

    def _apply(group: pd.DataFrame) -> pd.DataFrame:
        roll = group["mid"].rolling(window, min_periods=window)
        group = group.copy()
        group["tsfresh_abs_change"] = roll.apply(fc.absolute_sum_of_changes, raw=False)
        group["tsfresh_autocorr"] = roll.apply(lambda x: fc.autocorrelation(x, lag=1), raw=False)
        group["tsfresh_cid_ce"] = roll.apply(lambda x: fc.cid_ce(x, normalize=True), raw=False)
        group["tsfresh_kurtosis"] = roll.apply(fc.kurtosis, raw=False)
        group["tsfresh_skewness"] = roll.apply(fc.skewness, raw=False)
        return group

    if "Symbol" in df.columns:
        df = df.groupby("Symbol", group_keys=False).apply(_apply)
    else:
        df = _apply(df)

    return df
