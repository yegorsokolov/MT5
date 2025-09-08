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
fg_stub = types.ModuleType("analysis.feature_gate")
fg_stub.select = lambda df, tier, regime_id, persist=False: (df, [])
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
    capabilities=types.SimpleNamespace(
        cpus=8, memory_gb=16.0, has_gpu=False, gpu_count=0
    ),
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

spec = importlib.util.spec_from_file_location(
    "data.features", ROOT / "data" / "features.py"
)
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)


def test_add_cross_spectral_features_creates_columns():
    n = 70
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-01", periods=n, freq="D").tolist() * 2,
            "Symbol": ["AAA"] * n + ["BBB"] * n,
            "Close": np.random.rand(2 * n),
        }
    )
    out = features.add_cross_spectral_features(df, window=32)
    coh = out[out["Symbol"] == "AAA"]["coh_BBB"].dropna()
    assert not coh.empty
