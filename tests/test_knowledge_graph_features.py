import sys
import types
import importlib.util
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

features_pkg = types.ModuleType('features')
features_pkg.get_feature_pipeline = lambda: []
news_mod = types.ModuleType('features.news')
news_mod.add_economic_calendar_features = lambda df: df
news_mod.add_news_sentiment_features = lambda df: df
cross_asset_mod = types.ModuleType('features.cross_asset')
cross_asset_mod.add_index_features = lambda df: df
cross_asset_mod.add_cross_asset_features = lambda df: df
sys.modules['features'] = features_pkg
sys.modules['features.news'] = news_mod
sys.modules['features.cross_asset'] = cross_asset_mod

feature_gate_mod = types.ModuleType('analysis.feature_gate')
feature_gate_mod.select = lambda df, tier, regime_id, persist=False: (df, [])
sys.modules['analysis.feature_gate'] = feature_gate_mod

import analysis.knowledge_graph as kg

data_pkg = types.ModuleType("data")
data_pkg.__path__ = [str(repo_root / "data")]
sys.modules["data"] = data_pkg

spec_feat = importlib.util.spec_from_file_location("data.features", repo_root / "data" / "features.py")
feat = importlib.util.module_from_spec(spec_feat)
spec_feat.loader.exec_module(feat)


def test_add_knowledge_graph_features(monkeypatch):
    df = pd.DataFrame({"Symbol": ["A", "B"]})
    graph = object()

    monkeypatch.setattr(kg, "load_knowledge_graph", lambda path=None: graph)
    monkeypatch.setattr(feat, "risk_score", lambda g, c: {"A": 1.0, "B": 2.0}[c])
    monkeypatch.setattr(feat, "opportunity_score", lambda g, c: {"A": 3.0, "B": 4.0}[c])

    out = feat.add_knowledge_graph_features(df)
    assert list(out["graph_risk"]) == [1.0, 2.0]
    assert list(out["graph_opportunity"]) == [3.0, 4.0]


def test_make_features_includes_knowledge_graph(monkeypatch):
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2024", periods=2, freq="H"),
        "Symbol": ["A", "B"],
    })
    graph = object()

    monkeypatch.setattr(kg, "load_knowledge_graph", lambda path=None: graph)
    monkeypatch.setattr(feat, "risk_score", lambda g, c: {"A": 1.0, "B": 2.0}[c])
    monkeypatch.setattr(feat, "opportunity_score", lambda g, c: {"A": 3.0, "B": 4.0}[c])

    monkeypatch.setattr(feat, "aggregate_timeframes", lambda df, t: pd.DataFrame())
    monkeypatch.setattr(feat, "get_feature_pipeline", lambda: [])
    monkeypatch.setattr(feat, "add_garch_volatility", lambda df: df)
    monkeypatch.setattr(feat, "add_cross_spectral_features", lambda df: df)
    monkeypatch.setattr(feat, "add_frequency_features", lambda df: df)
    monkeypatch.setattr(feat, "add_fractal_features", lambda df: df)
    monkeypatch.setattr(feat.feature_gate, "select", lambda df, tier, regime, persist=False: (df, []))

    out = feat.make_features(df)
    assert list(out["graph_risk"]) == [1.0, 2.0]
    assert list(out["graph_opportunity"]) == [3.0, 4.0]
