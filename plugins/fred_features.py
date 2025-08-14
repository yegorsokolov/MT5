"""FRED economic data feature plugin.

min_cpus: 1
min_mem_gb: 1
requires_gpu: false
"""

from __future__ import annotations

MIN_CPUS = 1
MIN_MEM_GB = 1.0
REQUIRES_GPU = False

import logging
import pandas as pd
from typing import List

from . import register_feature
from utils import load_config

logger = logging.getLogger(__name__)


@register_feature
def add_fred_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge selected FRED time series into the feature dataframe."""
    cfg = load_config()
    if not cfg.get("use_fred_features", False):
        return df

    series: List[str] = cfg.get("fred_series", []) or []
    if not series:
        return df

    try:
        from pandas_datareader.data import DataReader
    except Exception as e:  # pragma: no cover - optional dependency
        logger.warning("pandas_datareader not available: %s", e)
        return df

    df = df.sort_values("Timestamp")
    start = pd.to_datetime(df["Timestamp"].min()).date()
    end = pd.to_datetime(df["Timestamp"].max()).date()

    for code in series:
        try:
            fred_df = DataReader(code, "fred", start, end)
        except Exception as e:  # pragma: no cover - runtime errors
            logger.warning("Failed to download %s from FRED: %s", code, e)
            df[f"fred_{code.lower()}"] = pd.NA
            continue

        fred_df = fred_df.rename(columns={code: f"fred_{code.lower()}"})
        fred_df = fred_df.sort_index()
        fred_df["date"] = pd.to_datetime(fred_df.index)
        df = pd.merge_asof(
            df,
            fred_df,
            left_on="Timestamp",
            right_on="date",
            direction="backward",
        ).drop(columns=["date"])

    return df
