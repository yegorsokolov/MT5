import sys
from pathlib import Path
import numpy as np
import pandas as pd
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Stub heavy dependencies to load data.features
features_stub = types.ModuleType("features")
features_stub.get_feature_pipeline = lambda: [lambda df: df]
news_stub = types.ModuleType("features.news")
news_stub.add_economic_calendar_features = lambda df: df
news_stub.add_news_sentiment_features = lambda df: df
cross_stub = types.ModuleType("features.cross_asset")
cross_stub.add_index_features = lambda df: df
cross_stub.add_cross_asset_features = lambda df: df
validators_stub = types.ModuleType("features.validators")
validators_stub.validate_ge = lambda df, *_: None
fg_stub = types.ModuleType("analysis.feature_gate")
fg_stub.select = lambda df, tier, regime_id, persist=False: (df, [])
adl_stub = types.ModuleType("analysis.data_lineage")
adl_stub.log_lineage = lambda *a, **k: None
analytics_stub = types.ModuleType("analytics")
metrics_stub = types.ModuleType("analytics.metrics_store")
metrics_stub.record_metric = lambda *a, **k: None
analytics_stub.metrics_store = metrics_stub
utils_stub = types.ModuleType("utils")
utils_stub.load_config = lambda: types.SimpleNamespace(features=types.SimpleNamespace(latency_threshold=0))
rm_stub = types.ModuleType("utils.resource_monitor")
rm_stub.ResourceCapabilities = types.SimpleNamespace
rm_stub.monitor = types.SimpleNamespace(
    capability_tier="lite",
    capabilities=types.SimpleNamespace(cpus=1, memory_gb=0.0, has_gpu=False, gpu_count=0),
    latency=lambda: 0.0,
)
config_stub = types.ModuleType("config_models")


class ConfigError(Exception):
    pass


config_stub.ConfigError = ConfigError
regime_stub = types.ModuleType("analysis.regime_detection")
regime_stub.periodic_reclassification = lambda df, step=500: df
fe_stub = types.ModuleType("analysis.feature_evolver")


class _Evolver:
    def apply_stored_features(self, df):
        return df

    def maybe_evolve(self, df, *args, **kwargs):
        return df


fe_stub.FeatureEvolver = _Evolver
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
        "features.validators": validators_stub,
        "analysis.feature_gate": fg_stub,
        "analysis.data_lineage": adl_stub,
        "analytics": analytics_stub,
        "analytics.metrics_store": metrics_stub,
        "utils": utils_stub,
        "utils.resource_monitor": rm_stub,
        "config_models": config_stub,
        "analysis.regime_detection": regime_stub,
        "analysis.feature_evolver": fe_stub,
        "data": data_pkg,
        "data.expectations": expect_stub,
        "data.multitimeframe": mtf_stub,
    }
)

import importlib.util
spec = importlib.util.spec_from_file_location("data.features", ROOT / "data" / "features.py")
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)

from analysis import frequency_features


def _synthetic_series(n=240, period=24):
    t = np.arange(n)
    seasonal = np.sin(2 * np.pi * t / period)
    trend = 0.1 * t
    series = seasonal + trend
    return series, seasonal, trend


def test_stl_decompose_recovers_components():
    series, seasonal, trend = _synthetic_series()
    res = frequency_features.stl_decompose(series, period=24)
    assert set(res.columns) == {"seasonal", "trend"}
    corr_seasonal = np.corrcoef(res["seasonal"][24:-24], seasonal[24:-24])[0, 1]
    corr_trend = np.corrcoef(res["trend"][24:-24], trend[24:-24])[0, 1]
    assert corr_seasonal > 0.9
    assert corr_trend > 0.9


def test_add_stl_features_adds_columns():
    series, seasonal, trend = _synthetic_series()
    df = pd.DataFrame({"mid": series})
    out = features.add_stl_features(df, period=24)
    assert {"stl_seasonal", "stl_trend"}.issubset(out.columns)
    corr_seasonal = np.corrcoef(out["stl_seasonal"][24:-24], seasonal[24:-24])[0, 1]
    corr_trend = np.corrcoef(out["stl_trend"][24:-24], trend[24:-24])[0, 1]
    assert corr_seasonal > 0.9
    assert corr_trend > 0.9
