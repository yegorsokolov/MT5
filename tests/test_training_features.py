from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mt5.config_models import AppConfig
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


def _bruteforce_best_threshold(y: np.ndarray, probs: np.ndarray) -> tuple[float, np.ndarray]:
    unique_scores = np.unique(probs)
    best_threshold = 0.5
    best_f1 = -1.0
    best_preds = np.zeros_like(y, dtype=int)

    for threshold in unique_scores:
        preds = (probs >= threshold).astype(int)
        tp = int(np.sum((preds == 1) & (y == 1)))
        fp = int(np.sum((preds == 1) & (y == 0)))
        fn = int(np.sum((preds == 0) & (y == 1)))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
            best_preds = preds

    return best_threshold, best_preds


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


def test_regime_thresholds_numpy_fallback_matches_bruteforce():
    import importlib

    module_name = "analysis.regime_thresholds"
    original_module = sys.modules.pop(module_name, None)
    original_sklearn = sys.modules.get("sklearn")
    original_metrics = sys.modules.get("sklearn.metrics")

    created_sklearn_stub = False
    if original_sklearn is None:
        sklearn_stub = types.ModuleType("sklearn")
        sklearn_stub.__path__ = []  # type: ignore[attr-defined]
        sys.modules["sklearn"] = sklearn_stub
        created_sklearn_stub = True

    broken_metrics = types.ModuleType("sklearn.metrics")

    def _missing_attr(name):  # pragma: no cover - exercised in tests
        raise ImportError("precision_recall_curve requires SciPy")

    broken_metrics.__getattr__ = _missing_attr  # type: ignore[attr-defined]
    sys.modules["sklearn.metrics"] = broken_metrics
    if created_sklearn_stub:
        sys.modules["sklearn"].metrics = broken_metrics  # type: ignore[attr-defined]

    try:
        regime_thresholds = importlib.import_module(module_name)
        pr_curve = regime_thresholds.find_regime_thresholds.__globals__["precision_recall_curve"]
        assert pr_curve.__module__ == module_name

        y = np.array([0, 1, 1, 0, 0, 1])
        probs = np.array([0.12, 0.81, 0.93, 0.25, 0.62, 0.78])
        regimes = np.array([0, 0, 0, 1, 1, 1])

        thresholds, preds = regime_thresholds.find_regime_thresholds(y, probs, regimes)

        expected_thresholds: dict[int, float] = {}
        expected_preds = np.zeros_like(y)
        for regime in np.unique(regimes):
            mask = regimes == regime
            thr, pred = _bruteforce_best_threshold(y[mask], probs[mask])
            expected_thresholds[int(regime)] = thr
            expected_preds[mask] = pred

        for regime, thr in expected_thresholds.items():
            assert np.isclose(thresholds[regime], thr)
        np.testing.assert_array_equal(preds, expected_preds)

        import math

        assert math.isfinite(sum(expected_thresholds.values()))
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module

        if original_metrics is not None:
            sys.modules["sklearn.metrics"] = original_metrics
        else:
            sys.modules.pop("sklearn.metrics", None)

        if created_sklearn_stub:
            sys.modules.pop("sklearn", None)
        elif original_sklearn is not None:
            sys.modules["sklearn"] = original_sklearn
