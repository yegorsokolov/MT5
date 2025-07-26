from . import register_feature
from utils import load_config
from regime import label_regimes
import pandas as pd

@register_feature
def add_hmm_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Add HMM based regime labels when enabled."""
    cfg = load_config()
    if not cfg.get("use_regime_classifier", False):
        return df

    n_states = cfg.get("regime_states", 3)
    return label_regimes(df, n_states=n_states, column="regime_hmm")
