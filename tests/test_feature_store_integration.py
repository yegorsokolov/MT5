import pandas as pd

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

import feature_store
import importlib.util
import types


class _RC:
    def __init__(self, cpus=1, memory_gb=0.0, has_gpu=False, gpu_count=0):
        self.cpus = cpus
        self.memory_gb = memory_gb
        self.has_gpu = has_gpu
        self.gpu_count = gpu_count


# stub dependencies to avoid heavy imports
features_pkg = types.ModuleType("features")
features_pkg.get_feature_pipeline = lambda: []
sys.modules["features"] = features_pkg
sys.modules["features.news"] = types.SimpleNamespace(
    add_economic_calendar_features=lambda df: df,
    add_news_sentiment_features=lambda df: df,
)
sys.modules["features.cross_asset"] = types.SimpleNamespace(
    add_index_features=lambda df: df,
    add_cross_asset_features=lambda df: df,
)
sys.modules["analytics"] = types.ModuleType("analytics")
sys.modules["analytics.metrics_store"] = types.SimpleNamespace(
    record_metric=lambda *a, **k: None
)
sys.modules["analysis"] = types.ModuleType("analysis")
sys.modules["analysis.cross_spectral"] = types.SimpleNamespace(
    compute=lambda d, window=1: d,
    REQUIREMENTS=_RC(),
)
sys.modules["analysis.knowledge_graph"] = types.SimpleNamespace(
    load_knowledge_graph=lambda: None,
    risk_score=lambda g, c: 0.0,
    opportunity_score=lambda g, c: 0.0,
)
sys.modules["analysis.feature_gate"] = types.SimpleNamespace(
    select=lambda df, tier, regime_id, persist=False: (df, [])
)
sys.modules["analysis.data_lineage"] = types.SimpleNamespace(
    log_lineage=lambda *a, **k: None
)
sys.modules["analysis.fractal_features"] = types.SimpleNamespace(
    rolling_fractal_features=lambda df: df
)
sys.modules["analysis.frequency_features"] = types.SimpleNamespace(
    spectral_features=lambda df: df,
    wavelet_energy=lambda df: df,
)
sys.modules["analysis.garch_vol"] = types.SimpleNamespace(
    garch_volatility=lambda df: df
)
utils_pkg = types.ModuleType("utils")
utils_pkg.load_config = lambda: {}
sys.modules["utils"] = utils_pkg
sys.modules["utils.resource_monitor"] = types.SimpleNamespace(
    monitor=types.SimpleNamespace(capabilities=_RC(), system_tier=lambda: "cpu"),
    ResourceCapabilities=_RC,
)
sys.modules["config_models"] = types.SimpleNamespace(ConfigError=Exception)

data_pkg = types.ModuleType("data")
data_pkg.__path__ = [str(repo_root / "data")]
sys.modules["data"] = data_pkg

sys.modules["data.expectations"] = types.SimpleNamespace(
    validate_dataframe=lambda df, name: None
)
sys.modules["data.multitimeframe"] = types.SimpleNamespace(
    aggregate_timeframes=lambda df, t: pd.DataFrame()
)

features_spec = importlib.util.spec_from_file_location(
    "data.features", repo_root / "data" / "features.py"
)
features = importlib.util.module_from_spec(features_spec)
features_spec.loader.exec_module(features)  # type: ignore[attr-defined]

river_module = types.ModuleType("river")
river_module.compose = types.SimpleNamespace(Pipeline=lambda *a, **k: None)
river_module.preprocessing = types.SimpleNamespace(StandardScaler=object)
river_module.linear_model = types.SimpleNamespace(LogisticRegression=object)
sys.modules["river"] = river_module
sys.modules["river.compose"] = river_module.compose
sys.modules["river.preprocessing"] = river_module.preprocessing
sys.modules["river.linear_model"] = river_module.linear_model

from mt5 import train_online


def test_offline_online_equivalence(tmp_path, monkeypatch):
    # use temporary directory for feature store
    monkeypatch.setattr(feature_store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(feature_store, "INDEX_FILE", tmp_path / "index.json")

    # avoid heavy multi-timeframe computation
    monkeypatch.setattr(features, "aggregate_timeframes", lambda df, t: pd.DataFrame())
    monkeypatch.setattr(features, "get_feature_pipeline", lambda: [])

    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "return": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    offline = features.make_features(df.copy())
    online = train_online.fetch_features()
    pd.testing.assert_frame_equal(offline, online)
