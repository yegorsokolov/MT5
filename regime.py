from __future__ import annotations

"""Market regime detection using a simple HMM."""

from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def label_regimes(df: pd.DataFrame, n_states: int = 3, column: str = "market_regime") -> pd.DataFrame:
    """Label each row with a market regime based on returns and volatility."""
    if "return" not in df.columns or "volatility_30" not in df.columns:
        return df.assign(**{column: 0})

    features = df[["return", "volatility_30"]].fillna(0).values
    try:
        model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100, random_state=42)
        model.fit(features)
        regimes = model.predict(features)
    except Exception:
        regimes = np.zeros(len(df), dtype=int)

    df[column] = regimes
    return df

