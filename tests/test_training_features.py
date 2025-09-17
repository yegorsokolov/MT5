from __future__ import annotations

from datetime import datetime, timedelta
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

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
        }
    )
    combined_signal = baseline_factor + order_flow_factor + rng.normal(scale=0.1, size=size)
    df["tb_label"] = (combined_signal > 0).astype(int)
    return df


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
