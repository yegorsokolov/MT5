import sys
from pathlib import Path
import types

import pandas as pd
import pytest

# Ensure lightweight data package import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
stub = types.ModuleType("data")
stub.__path__ = [str(DATA_ROOT)]
sys.modules.setdefault("data", stub)

utils_stub = types.ModuleType("utils")
utils_stub.load_config = lambda: {}
rm_stub = types.ModuleType("utils.resource_monitor")
rm_stub.ResourceCapabilities = types.SimpleNamespace
caps = types.SimpleNamespace(cpus=8, memory_gb=16, has_gpu=False, gpu_count=0)
rm_stub.monitor = types.SimpleNamespace(
    capability_tier="standard", capabilities=caps, subscribe=lambda: types.SimpleNamespace()
)
sys.modules.setdefault("utils", utils_stub)
sys.modules.setdefault("utils.resource_monitor", rm_stub)

analysis_stub = types.ModuleType("analysis")
analysis_stub.__path__ = [str(Path(__file__).resolve().parents[1] / "analysis")]
fg_stub = types.ModuleType("analysis.feature_gate")
fg_stub.select = lambda df, tier, regime_id, persist=False: (df, [])
sys.modules.setdefault("analysis", analysis_stub)
sys.modules.setdefault("analysis.feature_gate", fg_stub)

gym_stub = types.ModuleType("gym")
analytics_stub = types.ModuleType("analytics")
ms_stub = types.ModuleType("analytics.metrics_store")
ms_stub.record_metric = lambda *a, **k: None
ms_stub.TS_PATH = ""
analytics_stub.metrics_store = ms_stub
sys.modules["analytics"] = analytics_stub
sys.modules["analytics.metrics_store"] = ms_stub

import data.features as features  # type: ignore  # noqa: E402
from rl.trading_env import TradingEnv  # noqa: E402


def _patch_feature_pipeline(monkeypatch):
    """Stub out heavy feature dependencies for testing."""
    monkeypatch.setattr(features, "add_economic_calendar_features", lambda df: df)
    monkeypatch.setattr(features, "add_news_sentiment_features", lambda df: df)
    monkeypatch.setattr(features, "add_index_features", lambda df: df)
    monkeypatch.setattr(features, "add_cross_asset_features", lambda df: df)
    monkeypatch.setattr(features, "add_alt_features", lambda df: df)
    fund_stub = types.ModuleType("data.fundamental_loader")
    fund_stub.load_fundamental_data = lambda symbols: pd.DataFrame()
    sys.modules["data.fundamental_loader"] = fund_stub
    monkeypatch.setattr(features.feature_gate, "select", lambda df, tier, regime_id, persist=False: (df, df.columns))
    monkeypatch.setattr(features, "get_feature_pipeline", lambda: [lambda df: df])
    monkeypatch.setattr(features.monitor, "capability_tier", "standard")


def test_load_macro_features(monkeypatch):
    _patch_feature_pipeline(monkeypatch)
    df = pd.DataFrame(
        {
            "Timestamp": [pd.Timestamp("2020-01-15", tz="UTC"), pd.Timestamp("2020-02-15", tz="UTC")],
            "Symbol": ["AAA", "AAA"],
            "Region": ["US", "US"],
            "Bid": [1.0, 1.1],
            "Ask": [1.01, 1.11],
        }
    )
    out = features.make_features(df)
    assert "macro_gdp" in out.columns
    assert list(out["macro_gdp"]) == [pytest.approx(1.0), pytest.approx(1.1)]
    assert "macro_cpi" in out.columns


def test_env_exposes_macro_features():
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-01", periods=3, freq="D"),
            "Symbol": ["AAA"] * 3,
            "mid": [1.0, 1.1, 1.2],
            "feat": [0.1, 0.2, 0.3],
            "macro_gdp": [1.0, 1.0, 1.0],
        }
    )
    env = TradingEnv(df, features=["feat"], macro_features=["macro_gdp"])
    obs = env.reset()
    assert obs.shape[0] == 2
    assert obs[-1] == pytest.approx(1.0)
    nxt, _, _, _ = env.step({"size": [0.0], "close": [0.0]})
    assert nxt[-1] == pytest.approx(1.0)
