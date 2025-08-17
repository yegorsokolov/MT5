import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from analysis.feature_evolver import FeatureEvolver


def test_feature_evolver(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "a": rng.normal(size=50),
        "b": rng.normal(size=50),
    })
    df["return"] = df["a"] * 0.5 + df["b"] * 0.2
    df["market_regime"] = 1

    evolver = FeatureEvolver(store_dir=tmp_path)
    df2 = evolver.maybe_evolve(df.copy(), target_col="return")
    # ensure a new feature column was added
    gp_cols = [c for c in df2.columns if c.startswith("gp_feat_")]
    assert gp_cols, "no evolved features created"

    # ensure we can load and apply stored features on new data
    df_new = df.copy()
    df_new = evolver.apply_stored_features(df_new)
    for col in gp_cols:
        assert col in df_new.columns
