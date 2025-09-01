import sys

import numpy as np
import pandas as pd

# Remove scipy stubs installed by the global test configuration so that
# downstream imports (e.g., scikit-learn) can access the real SciPy package.
sys.modules.pop("scipy", None)
sys.modules.pop("scipy.stats", None)
sys.modules.pop("pandas", None)

from analysis.robust_filters import median_filter, trimmed_mean_filter, zscore_clamp
from data.features import make_features


def test_zscore_clamp_basic():
    s = pd.Series([1] * 10 + [1000])
    clamped, idx = zscore_clamp(s, threshold=3)
    assert idx  # anomaly detected
    assert clamped.iloc[-1] != 1000
    mu, sigma = s.mean(), s.std(ddof=0)
    zscores = (clamped - mu) / sigma
    assert (zscores.abs() <= 3.0 + 1e-8).all()


def test_features_consistency_with_outliers():
    timestamps = pd.date_range("2020", periods=50, freq="S")
    np.random.seed(0)
    base_bid = np.linspace(100, 101, 50)
    base_ask = base_bid + 0.1
    df_clean = pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Bid": base_bid,
            "Ask": base_ask,
            "BidVolume": np.random.randint(1, 5, size=50),
            "AskVolume": np.random.randint(1, 5, size=50),
            "Symbol": "TEST",
        }
    )

    features_clean = make_features(df_clean)

    df_dirty = df_clean.copy()
    df_dirty.loc[10, "Bid"] = 1000
    df_dirty.loc[20, "Ask"] = -1000
    features_dirty = make_features(df_dirty)

    num_cols = features_clean.select_dtypes(include=[np.number]).columns
    assert np.allclose(
        features_clean[num_cols].to_numpy(),
        features_dirty[num_cols].to_numpy(),
        atol=0.02,
        equal_nan=True,
    )
