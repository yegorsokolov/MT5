"""Regime feature plugin providing stable basket identifiers.

min_cpus: 1
min_mem_gb: 0.5
requires_gpu: false
"""

MIN_CPUS = 1
MIN_MEM_GB = 0.5
REQUIRES_GPU = False

from . import register_feature
from utils import load_config
from mt5.regime import label_regimes
import pandas as pd


@register_feature
def add_hmm_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime basket labels when enabled."""
    cfg = load_config()
    if not cfg.get("use_regime_classifier", False):
        return df

    n_states = cfg.get("regime_states", 3)
    method = cfg.get("regime_method", "kmeans")
    return label_regimes(df, n_states=n_states, column="regime_hmm", method=method)
