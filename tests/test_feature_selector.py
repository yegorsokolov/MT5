import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from analysis.feature_selector import select_features
from mt5.config_models import AppConfig
from mt5.train_utils import resolve_training_features


def _make_cfg(training: dict | None = None) -> AppConfig:
    cfg_dict: dict = {
        "strategy": {"symbols": ["TEST"], "risk_per_trade": 0.01},
    }
    if training:
        cfg_dict["training"] = training
    return AppConfig.model_validate(cfg_dict)


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


def test_resolve_training_features_respects_includes_and_excludes():
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=256)
    x2 = x1 * 0.5 + rng.normal(scale=0.1, size=256)
    df = pd.DataFrame({"x1": x1, "x2": x2})
    y = (x1 + rng.normal(scale=0.2, size=256) > 0).astype(int)

    cfg = _make_cfg(
        {
            "feature_includes": ["x2"],
            "feature_excludes": ["x1"],
            "use_feature_selector": False,
        }
    )
    features = resolve_training_features(df, pd.Series(y, name="target"), cfg)
    assert "x2" in features
    assert "x1" not in features


def test_resolve_training_features_family_overrides():
    rng = np.random.default_rng(0)
    cross = rng.normal(size=256)
    base = rng.normal(size=256)
    df = pd.DataFrame({
        "cross_corr_SP500": cross,
        "return": base,
    })
    y = (base + rng.normal(scale=0.1, size=256) > 0).astype(int)

    drop_cfg = _make_cfg(
        {
            "feature_families": {"cross_asset": False},
            "use_feature_selector": False,
        }
    )
    keep_cfg = _make_cfg(
        {
            "feature_includes": ["cross_corr_SP500"],
            "feature_families": {"cross_asset": False},
            "use_feature_selector": False,
        }
    )

    dropped = resolve_training_features(df, pd.Series(y, name="target"), drop_cfg)
    kept = resolve_training_features(df, pd.Series(y, name="target"), keep_cfg)

    assert "cross_corr_SP500" not in dropped
    assert "cross_corr_SP500" in kept
