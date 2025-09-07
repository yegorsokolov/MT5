import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import types

# Ensure project root on path for direct execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Stub heavy modules before importing data.features
features_stub = types.ModuleType("features")
features_stub.get_feature_pipeline = lambda: [lambda df: df]
news_stub = types.ModuleType("features.news")
news_stub.add_economic_calendar_features = lambda df: df
news_stub.add_news_sentiment_features = lambda df: df
cross_stub = types.ModuleType("features.cross_asset")
cross_stub.add_index_features = lambda df: df
cross_stub.add_cross_asset_features = lambda df: df
analysis_stub = types.ModuleType("analysis")
fg_stub = types.ModuleType("analysis.feature_gate")
fg_stub.select = lambda df, tier, regime_id, persist=False: (df, [])
analysis_stub.feature_gate = fg_stub
adl_stub = types.ModuleType("analysis.data_lineage")
adl_stub.log_lineage = lambda *a, **k: None
dq_stub = types.ModuleType("analysis.data_quality")
dq_stub.apply_quality_checks = lambda df: df
analytics_stub = types.ModuleType("analytics")
metrics_stub = types.ModuleType("analytics.metrics_store")
metrics_stub.record_metric = lambda *a, **k: None
analytics_stub.metrics_store = metrics_stub
utils_stub = types.ModuleType("utils")
utils_stub.load_config = lambda: {}
rm_stub = types.ModuleType("utils.resource_monitor")
rm_stub.ResourceCapabilities = types.SimpleNamespace
rm_stub.monitor = types.SimpleNamespace(
    capability_tier="lite",
    capabilities=types.SimpleNamespace(cpus=1, memory_gb=0.0, has_gpu=False, gpu_count=0),
    latency=lambda: 0.0,
)
data_backend_stub = types.ModuleType("utils.data_backend")
data_backend_stub.get_dataframe_module = lambda: pd
data_pkg = types.ModuleType("data")
data_pkg.__path__ = [str(ROOT / "data")]
expect_stub = types.ModuleType("data.expectations")
expect_stub.validate_dataframe = lambda df, *_: None
mtf_stub = types.ModuleType("data.multitimeframe")
mtf_stub.aggregate_timeframes = lambda df, tfs: pd.DataFrame()
sys.modules.update(
    {
        "features": features_stub,
        "features.news": news_stub,
        "features.cross_asset": cross_stub,
        "analysis": analysis_stub,
        "analysis.feature_gate": fg_stub,
        "analysis.data_lineage": adl_stub,
        "analysis.data_quality": dq_stub,
        "analytics": analytics_stub,
        "analytics.metrics_store": metrics_stub,
        "utils": utils_stub,
        "utils.resource_monitor": rm_stub,
        "utils.data_backend": data_backend_stub,
        "data": data_pkg,
        "data.expectations": expect_stub,
        "data.multitimeframe": mtf_stub,
    }
)

import importlib.util
spec = importlib.util.spec_from_file_location("data.features", ROOT / "data" / "features.py")
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)


def test_add_time_features_values():
    times = pd.to_datetime([
        "2024-01-01T00:00Z",  # Monday midnight
        "2024-01-03T12:00Z",  # Wednesday noon
    ])
    df = pd.DataFrame({"Timestamp": times})
    out = features.add_time_features(df)

    assert np.isclose(out.loc[0, "hour_of_day_sin"], 0.0)
    assert np.isclose(out.loc[0, "hour_of_day_cos"], 1.0)
    assert np.isclose(out.loc[0, "day_of_week_sin"], 0.0)
    assert np.isclose(out.loc[0, "day_of_week_cos"], 1.0)

    assert np.isclose(out.loc[1, "hour_of_day_sin"], 0.0, atol=1e-6)
    assert np.isclose(out.loc[1, "hour_of_day_cos"], -1.0, atol=1e-6)
    angle = 2 * np.pi * 2 / 7  # Wednesday
    assert np.isclose(out.loc[1, "day_of_week_sin"], np.sin(angle))
    assert np.isclose(out.loc[1, "day_of_week_cos"], np.cos(angle))


def test_make_features_includes_time_features(monkeypatch):
    monkeypatch.setattr(features, "aggregate_timeframes", lambda df, tfs: pd.DataFrame())
    monkeypatch.setattr(features, "get_feature_pipeline", lambda: [lambda df: df])
    monkeypatch.setattr(
        features.feature_gate, "select", lambda df, tier, regime_id, persist=False: (df, [])
    )
    features.monitor.capability_tier = "lite"
    features.monitor.capabilities = features.ResourceCapabilities(
        cpus=1, memory_gb=0.0, has_gpu=False, gpu_count=0
    )

    base = pd.DataFrame({"Timestamp": pd.date_range("2024-01-01", periods=2, freq="H", tz="UTC")})
    out = features.make_features(base)
    cols = {
        "hour_of_day_sin",
        "hour_of_day_cos",
        "day_of_week_sin",
        "day_of_week_cos",
    }
    assert cols.issubset(out.columns)
