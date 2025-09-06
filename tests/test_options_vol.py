import sys
import types
from pathlib import Path

import pandas as pd

# Stub utils and analysis modules to avoid heavy dependencies during import
data_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(data_root))

# Provide a lightweight stub for the data package to avoid importing heavy
# dependencies from ``data.__init__`` during tests.
data_stub = types.ModuleType("data")
data_stub.__path__ = [str(data_root / "data")]
sys.modules.setdefault("data", data_stub)

# Stub external feature modules
features_stub = types.ModuleType("features")
features_stub.get_feature_pipeline = lambda: [lambda df: df]
news_stub = types.ModuleType("features.news")
news_stub.add_economic_calendar_features = lambda df: df
news_stub.add_news_sentiment_features = lambda df: df
cross_stub = types.ModuleType("features.cross_asset")
cross_stub.add_index_features = lambda df: df
cross_stub.add_cross_asset_features = lambda df: df
sys.modules.setdefault("features", features_stub)
sys.modules.setdefault("features.news", news_stub)
sys.modules.setdefault("features.cross_asset", cross_stub)

utils_stub = types.ModuleType("utils")
utils_stub.load_config = lambda: {}
rm_stub = types.ModuleType("utils.resource_monitor")
rm_stub.ResourceCapabilities = types.SimpleNamespace
rm_stub.monitor = types.SimpleNamespace(
    capability_tier="lite",
    capabilities=types.SimpleNamespace(cpus=1, memory_gb=1, has_gpu=False, gpu_count=0),
    subscribe=lambda: types.SimpleNamespace(),
)
sys.modules.setdefault("utils", utils_stub)
sys.modules.setdefault("utils.resource_monitor", rm_stub)

analysis_stub = types.ModuleType("analysis")
analysis_stub.__path__ = [str(data_root / "analysis")]
fg_stub = types.ModuleType("analysis.feature_gate")
fg_stub.select = lambda df, tier, regime_id, persist=False: (df, df.columns)
sys.modules.setdefault("analysis", analysis_stub)
sys.modules.setdefault("analysis.feature_gate", fg_stub)

import data.features as feat
from data import options_vol, vol_skew
from utils.resource_monitor import ResourceCapabilities, monitor
import pytest


def _sample_options_df():
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "implied_vol": [0.2, 0.25],
            "vol_skew": [0.1, 0.15],
        }
    )


def test_options_vol_compute(monkeypatch):
    df = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(["2024-01-03", "2024-01-04"], utc=True),
            "Symbol": ["ABC", "ABC"],
        }
    )

    monkeypatch.setattr(options_vol, "_read_local_csv", lambda sym: _sample_options_df())
    res = options_vol.compute(df.copy())
    assert res["implied_vol"].tolist() == [pytest.approx(0.25), pytest.approx(0.25)]
    assert res["vol_skew"].tolist() == [pytest.approx(0.15), pytest.approx(0.15)]


def test_make_features_merges_options(monkeypatch):
    df = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(["2024-01-03", "2024-01-04"], utc=True),
            "Symbol": ["ABC", "ABC"],
        }
    )

    monkeypatch.setattr(options_vol, "_read_local_csv", lambda sym: _sample_options_df())

    # Avoid heavy feature pipeline
    monkeypatch.setattr(feat, "get_feature_pipeline", lambda: [lambda x: x])

    # Disable cross spectral by requiring unrealistic resources
    from analysis import cross_spectral

    monkeypatch.setattr(
        cross_spectral,
        "REQUIREMENTS",
        ResourceCapabilities(cpus=999, memory_gb=999, has_gpu=False, gpu_count=0),
    )

    # Lower requirements for options modules for the test
    monkeypatch.setattr(
        options_vol, "REQUIREMENTS", ResourceCapabilities(cpus=1, memory_gb=0, has_gpu=False, gpu_count=0)
    )
    monkeypatch.setattr(
        vol_skew, "REQUIREMENTS", ResourceCapabilities(cpus=1, memory_gb=0, has_gpu=False, gpu_count=0)
    )

    # Provide capabilities to trigger options merge
    monkeypatch.setattr(
        monitor, "capabilities", ResourceCapabilities(cpus=8, memory_gb=8, has_gpu=False, gpu_count=0)
    )
    monkeypatch.setattr(monitor, "capability_tier", "lite")

    res = feat.make_features(df.copy())
    assert res["implied_vol"].tolist() == [pytest.approx(0.25), pytest.approx(0.25)]
    assert res["vol_skew"].tolist() == [pytest.approx(0.15), pytest.approx(0.15)]
