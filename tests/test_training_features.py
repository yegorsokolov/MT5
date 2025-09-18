from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config_models import AppConfig
from training.features import (
    append_risk_profile_features,
    build_feature_candidates,
    ensure_mandatory_features,
    select_model_features,
)


class _RiskProfile:
    def __init__(self, tolerance: float = 0.5) -> None:
        self.leverage_cap = 2.0
        self.drawdown_limit = 0.1
        self.tolerance = tolerance


def _make_df(size: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    baseline_factor = rng.normal(size=size)
    order_flow_factor = rng.normal(size=size)
    ts = [datetime(2020, 1, 1) + timedelta(minutes=i) for i in range(size)]
    df = pd.DataFrame(
        {
            "Timestamp": ts,
            "Symbol": ["TEST"] * size,
            "return": baseline_factor,
            "ma_5": baseline_factor + rng.normal(scale=0.01, size=size),
            "imbalance": order_flow_factor + rng.normal(scale=0.05, size=size),
            "cvd": order_flow_factor * 0.5 + rng.normal(scale=0.05, size=size),
            "coh_EURUSD": rng.normal(scale=0.1, size=size),
            "cross_corr_SP500": rng.normal(scale=0.1, size=size),
            "volume_ratio": rng.normal(scale=0.05, size=size),
            "volume_imbalance": rng.normal(scale=0.05, size=size),
        }
    )
    combined_signal = baseline_factor + order_flow_factor + rng.normal(scale=0.1, size=size)
    df["tb_label"] = (combined_signal > 0).astype(int)
    return df


def _make_config(training: dict | None = None) -> AppConfig:
    cfg_dict: dict = {
        "strategy": {"symbols": ["TEST"], "risk_per_trade": 0.01},
    }
    if training:
        cfg_dict["training"] = training
    return AppConfig.model_validate(cfg_dict)


def test_append_risk_profile_features_adds_budget_columns():
    df = _make_df()
    profile = _RiskProfile()
    budget = append_risk_profile_features(df, profile)
    for key in budget.as_features().keys():
        assert key in df.columns
    assert "risk_tolerance" in df.columns


def test_build_and_select_features_respects_mandatory_columns():
    df = _make_df()
    profile = _RiskProfile()
    append_risk_profile_features(df, profile)
    candidates = build_feature_candidates(df, None)
    mandatory = ensure_mandatory_features(["risk_tolerance"], ["risk_tolerance"])
    selected = select_model_features(
        df,
        candidates,
        df["tb_label"],
        model_type="lgbm",
        mandatory=mandatory,
    )
    assert "risk_tolerance" in selected


def test_select_model_features_handles_cross_modal():
    df = _make_df()
    df["price_window_0"] = np.random.random(len(df))
    df["news_emb_0"] = np.random.random(len(df))
    candidates = ["price_window_0", "news_emb_0"]
    selected = select_model_features(
        df,
        candidates,
        df["tb_label"],
        model_type="cross_modal",
        mandatory=None,
    )
    assert set(selected) == set(candidates)


def test_build_feature_candidates_honors_config_overrides():
    df = _make_df()
    profile = _RiskProfile()
    budget = append_risk_profile_features(df, profile)
    cfg = _make_config(
        {
            "feature_includes": ["imbalance"],
            "feature_excludes": ["return", "ma_5"],
            "feature_families": {"order_flow": True},
        }
    )
    candidates = build_feature_candidates(df, budget, cfg=cfg)
    assert "imbalance" in candidates
    assert "cvd" in candidates  # family include pulls in the rest
    assert "return" not in candidates
    assert "ma_5" not in candidates


def test_build_feature_candidates_family_exclusion():
    df = _make_df()
    cfg = _make_config({"feature_families": {"cross_asset": False}})
    candidates = build_feature_candidates(df, None, cfg=cfg)
    assert all(not col.startswith("cross_corr_") for col in candidates)


def test_build_feature_candidates_custom_groups():
    df = _make_df()
    df["macro_spread"] = np.random.normal(size=len(df))
    cfg_enable = _make_config(
        {
            "feature_groups": {"macro": ["macro_spread"]},
            "feature_families": {"macro": True},
        }
    )
    cfg_disable = _make_config(
        {
            "feature_groups": {"macro": ["macro_spread"]},
            "feature_families": {"macro": False},
        }
    )
    enabled = build_feature_candidates(df, None, cfg=cfg_enable)
    disabled = build_feature_candidates(df, None, cfg=cfg_disable)
    assert "macro_spread" in enabled
    assert "macro_spread" not in disabled
