import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from analysis.feature_selector import select_features


def test_select_features_drops_noise_and_preserves_predictions(tmp_path):
    rng = np.random.default_rng(0)
    signal = rng.normal(size=200)
    noise = rng.normal(size=200)
    y = (signal > 0).astype(int)
    df = pd.DataFrame({"signal": signal, "noise": noise})

    selected = select_features(df, y)
    assert "signal" in selected
    assert "noise" not in selected

    # Persist and reload feature list to ensure stability for inference
    feat_path = tmp_path / "selected_features.json"
    feat_path.write_text(json.dumps(selected))
    loaded = json.loads(feat_path.read_text())
    assert loaded == selected


def test_correlated_features_are_removed():
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=200)
    x2 = x1 * 0.95 + rng.normal(scale=0.1, size=200)
    y = (x1 + x2 + rng.normal(scale=0.1, size=200) > 0).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2})

    selected_no_filter = select_features(df, y, corr_threshold=1.0)
    assert set(["x1", "x2"]).issubset(selected_no_filter)

    selected = select_features(df, y, corr_threshold=0.9)
    assert len([f for f in ["x1", "x2"] if f in selected]) == 1
