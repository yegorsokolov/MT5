import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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

    model_selected = LogisticRegression().fit(df[selected], y)
    preds_sel = model_selected.predict(df[selected])

    # Training only on the true signal should yield identical predictions
    model_true = LogisticRegression().fit(df[["signal"]], y)
    preds_true = model_true.predict(df[["signal"]])
    assert np.array_equal(preds_sel, preds_true)

    # Persist and reload feature list to ensure stability for inference
    feat_path = tmp_path / "selected_features.json"
    feat_path.write_text(json.dumps(selected))
    loaded = json.loads(feat_path.read_text())
    assert loaded == selected
    assert np.array_equal(model_selected.predict(df[loaded]), preds_sel)
