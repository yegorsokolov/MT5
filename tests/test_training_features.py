from __future__ import annotations

from datetime import datetime, timedelta
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from train_utils import resolve_training_features


def _make_df(size: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    baseline_factor = rng.normal(size=size)
    order_flow_factor = rng.normal(size=size)
    ts = [datetime(2020, 1, 1) + timedelta(minutes=i) for i in range(size)]
    df = pd.DataFrame(
        {
            "Timestamp": ts,
            "Symbol": ["TEST"] * size,
            "baseline_signal": baseline_factor + rng.normal(scale=0.05, size=size),
            "long_stop": rng.normal(scale=0.1, size=size),
            "imbalance": order_flow_factor + rng.normal(scale=0.05, size=size),
            "cvd": order_flow_factor * 0.5 + rng.normal(scale=0.05, size=size),
            "coh_EURUSD": rng.normal(scale=0.1, size=size),
            "noise": rng.normal(size=size),
        }
    )
    combined_signal = baseline_factor + order_flow_factor + rng.normal(scale=0.1, size=size)
    df["tb_label"] = (combined_signal > 0).astype(int)
    return df


def test_resolve_training_features_includes_predictive_families():
    df = _make_df()
    cfg: dict[str, object] = {}
    features = resolve_training_features(df, df["tb_label"], cfg)

    assert "Timestamp" not in features
    assert "Symbol" not in features
    assert "baseline_signal" in features
    assert any(col in features for col in ("imbalance", "cvd"))


def test_order_flow_family_can_be_removed_via_config():
    df = _make_df()
    cfg = {"feature_families": {"order_flow": False}}
    features = resolve_training_features(df, df["tb_label"], cfg)

    assert "imbalance" not in features
    assert "cvd" not in features


def test_cross_spectral_family_can_be_forced():
    df = _make_df()
    df["coh_EURUSD"] = 0.0  # not predictive on its own
    cfg = {"feature_families": {"cross_spectral": True}}
    features = resolve_training_features(df, df["tb_label"], cfg)

    assert "coh_EURUSD" in features


def test_mandatory_features_are_preserved():
    df = _make_df()
    df["risk_tolerance"] = 0.5
    cfg: dict[str, object] = {}
    features = resolve_training_features(
        df,
        df["tb_label"],
        cfg,
        mandatory=["risk_tolerance"],
    )

    assert "risk_tolerance" in features
