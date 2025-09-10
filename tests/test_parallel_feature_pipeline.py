import time
import types
import sys

import pandas as pd

import importlib
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("requests", types.SimpleNamespace())

from utils.resource_monitor import ResourceCapabilities


def _fast_df():
    return pd.DataFrame({"Timestamp": pd.date_range("2021", periods=2, freq="H")})


def _stub(df):
    return df


def test_parallel_feature_pipeline(monkeypatch):
    pkg = types.ModuleType("data")
    pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "data")]
    sys.modules["data"] = pkg

    sys.modules["data.expectations"] = types.SimpleNamespace(validate_dataframe=lambda df, **k: df)

    features_stub = types.ModuleType("features")
    features_stub.get_feature_pipeline = lambda: []
    features_stub.news = types.SimpleNamespace(
        add_economic_calendar_features=lambda df: df,
        add_news_sentiment_features=lambda df: df,
    )
    features_stub.cross_asset = types.SimpleNamespace(
        add_index_features=lambda df: df,
        add_cross_asset_features=lambda df: df,
    )
    sys.modules["features"] = features_stub
    sys.modules["features.news"] = features_stub.news
    sys.modules["features.cross_asset"] = features_stub.cross_asset

    kg_mod = types.ModuleType("analysis.knowledge_graph")
    kg_mod.load_knowledge_graph = lambda: None
    kg_mod.risk_score = lambda df, graph=None: df
    kg_mod.opportunity_score = lambda df, graph=None: df
    sys.modules["analysis.knowledge_graph"] = kg_mod

    cs_mod = types.ModuleType("analysis.cross_spectral")
    cs_mod.compute = lambda df, window=None: df
    cs_mod.REQUIREMENTS = ResourceCapabilities(0, 0, False, 0)
    sys.modules["analysis.cross_spectral"] = cs_mod

    sys.modules["analysis.fractal_features"] = types.SimpleNamespace(rolling_fractal_features=lambda df: df)
    sys.modules["analysis.frequency_features"] = types.SimpleNamespace(
        spectral_features=lambda df: df, wavelet_energy=lambda df: df
    )
    sys.modules["analysis.garch_vol"] = types.SimpleNamespace(garch_volatility=lambda df: df)
    sys.modules["analysis.feature_gate"] = types.SimpleNamespace(select=lambda df, *a, **k: (df, []))
    sys.modules["analysis.data_lineage"] = types.SimpleNamespace(log_lineage=lambda *a, **k: None)
    sys.modules["great_expectations"] = types.SimpleNamespace()

    feat = importlib.import_module("data.features")

    # Skip heavy internal features
    monkeypatch.setattr(feat, "aggregate_timeframes", lambda df, tfs: pd.DataFrame())
    monkeypatch.setattr(feat, "add_time_features", _stub)
    monkeypatch.setattr(feat, "add_garch_volatility", _stub)
    monkeypatch.setattr(feat, "add_cross_spectral_features", _stub)
    monkeypatch.setattr(feat, "add_knowledge_graph_features", _stub)
    monkeypatch.setattr(feat, "add_frequency_features", _stub)
    monkeypatch.setattr(feat, "add_fractal_features", _stub)
    monkeypatch.setattr(
        feat,
        "monitor",
        types.SimpleNamespace(capabilities=ResourceCapabilities(0, 0, False, 0), capability_tier="lite"),
        raising=False,
    )
    sys.modules['dataset'] = types.SimpleNamespace(FEATURE_PLUGINS=[])

    def f1(df):
        time.sleep(0.2)
        df = df.copy(); df['f1'] = 1; return df
    def f2(df):
        time.sleep(0.2)
        df = df.copy(); df['f2'] = 1; return df
    def f3(df):
        time.sleep(0.2)
        df = df.copy(); df['f3'] = 1; return df

    pipeline = [f1, f2, f3]
    monkeypatch.setattr(feat, "get_feature_pipeline", lambda: pipeline)

    df = _fast_df()
    start_parallel = time.perf_counter()
    out_parallel = feat.make_features(df.copy())
    parallel_time = time.perf_counter() - start_parallel

    start_seq = time.perf_counter()
    df_seq = df.copy()
    for func in pipeline:
        df_seq = func(df_seq)
    seq_time = time.perf_counter() - start_seq

    assert list(out_parallel.columns[:4]) == ["Timestamp", "f1", "f2", "f3"]
    assert parallel_time < seq_time
    assert parallel_time < 0.5
