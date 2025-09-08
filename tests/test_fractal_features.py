import sys
from pathlib import Path
import numpy as np
import pandas as pd
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.fractal_features import (
    rolling_fractal_dimension,
    rolling_hurst_exponent,
)

# Stub heavy dependencies for importing data.features
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
        "analytics": analytics_stub,
        "analytics.metrics_store": metrics_stub,
        "utils": utils_stub,
        "utils.resource_monitor": rm_stub,
        "data": data_pkg,
        "data.expectations": expect_stub,
        "data.multitimeframe": mtf_stub,
    }
)

import importlib.util
spec = importlib.util.spec_from_file_location("data.features", ROOT / "data" / "features.py")
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)


def test_trend_series_has_high_hurst_low_dimension():
    n = 200
    series = pd.Series(np.linspace(0, 1, n))
    hurst = rolling_hurst_exponent(series, window=n).iloc[-1]
    fd = rolling_fractal_dimension(series, window=n).iloc[-1]
    assert hurst > 0.9
    assert fd < 1.1


def test_random_walk_series_hurst_near_half_dimension_high():
    np.random.seed(0)
    n = 400
    series = pd.Series(np.cumsum(np.random.randn(n)))
    hurst = rolling_hurst_exponent(series, window=n).iloc[-1]
    fd = rolling_fractal_dimension(series, window=n).iloc[-1]
    assert 0.4 < hurst < 0.7
    assert 1.3 < fd < 2.0


def test_add_fractal_features_shape_and_non_null():
    n = 200
    df = pd.DataFrame({"mid": np.linspace(1.0, 2.0, n)})
    out = features.add_fractal_features(df)
    assert {"hurst", "fractal_dim"}.issubset(out.columns)
    assert out[["hurst", "fractal_dim"]].shape == (n, 2)
    assert pd.notnull(out.loc[n - 1, "hurst"])
    assert pd.notnull(out.loc[n - 1, "fractal_dim"])
