import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.modules.pop("scipy", None)
sys.modules.pop("scipy.stats", None)
import scipy  # ensure real scipy is loaded
from analysis.feature_evolver import FeatureEvolver


def test_feature_evolver(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=80), "b": rng.normal(size=80)})
    df["return"] = df["a"] * df["b"] + rng.normal(scale=0.1, size=80)
    df["market_regime"] = 1

    store = tmp_path / "store"
    module_path = tmp_path / "features_mod.py"
    evolver = FeatureEvolver(store_dir=store)

    df2 = evolver.maybe_evolve(
        df.copy(),
        target_col="return",
        module_path=module_path,
        generations=5,
        population_size=100,
    )
    gp_cols = [c for c in df2.columns if c.startswith("gp_feat_")]
    assert gp_cols, "no evolved features created"

    # manifest should log expression and score
    manifest_text = (store / "manifest.json").read_text()
    manifest = json.loads(manifest_text)
    assert manifest[0]["expression"]
    assert isinstance(manifest[0]["score"], float)

    # running again should yield identical manifest and not duplicate file output
    file_text = module_path.read_text()
    evolver.maybe_evolve(df.copy(), target_col="return", module_path=module_path)
    assert manifest_text == (store / "manifest.json").read_text()
    assert file_text == module_path.read_text()

    # ensure we can load and apply stored features on new data
    df_new = evolver.apply_stored_features(df.copy())
    for col in gp_cols:
        assert col in df_new.columns
