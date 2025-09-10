import importlib.util
import types
import sys
from pathlib import Path
from functools import wraps

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def _setup_features_stub():
    features_pkg = types.ModuleType("features")
    features_pkg.__path__ = [str(ROOT / "features")]

    def validate_module(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from data.expectations import validate_dataframe
            result = func(*args, **kwargs)
            suite = func.__module__.split(".")[-1]
            validate_dataframe(result, suite)
            return result
        return wrapper

    features_pkg.validate_module = validate_module
    sys.modules["features"] = features_pkg


def _stub_utils():
    util_mod = types.ModuleType("utils")
    util_mod.load_config = lambda: {}
    sys.modules["utils"] = util_mod

    res_mod = types.ModuleType("utils.resource_monitor")
    class RC:
        def __init__(self, cpus=0, memory_gb=0.0, has_gpu=False, gpu_count=0):
            self.cpus = cpus
            self.memory_gb = memory_gb
            self.has_gpu = has_gpu
            self.gpu_count = gpu_count
    res_mod.ResourceCapabilities = RC
    res_mod.monitor = types.SimpleNamespace(capabilities=RC(), subscribe=lambda: types.SimpleNamespace())
    sys.modules["utils.resource_monitor"] = res_mod


def _stub_expectations(called):
    stub = types.ModuleType("data.expectations")

    def _validate(df, suite, **_):
        called["suite"] = suite
        called["columns"] = list(df.columns)
        raise ValueError("failed")

    stub.validate_dataframe = _validate
    sys.modules["data.expectations"] = stub
    data_pkg = sys.modules.setdefault("data", types.ModuleType("data"))
    data_pkg.expectations = stub


def _stub_data_features():
    df_mod = types.ModuleType("data.features")

    def add_economic_calendar_features(df):
        df = df.copy()
        df["minutes_to_event"] = 0
        return df

    def add_news_sentiment_features(df):
        df = df.copy()
        df["news_sentiment"] = 0
        return df

    df_mod.add_economic_calendar_features = add_economic_calendar_features
    df_mod.add_news_sentiment_features = add_news_sentiment_features
    data_pkg = sys.modules.setdefault("data", types.ModuleType("data"))
    data_pkg.features = df_mod
    sys.modules["data.features"] = df_mod
    events_mod = types.ModuleType("data.events")
    events_mod.get_events = lambda past_events=True: []
    data_pkg.events = events_mod
    sys.modules["data.events"] = events_mod


def _import_feature_module(name: str):
    spec = importlib.util.spec_from_file_location(f"features.{name}", ROOT / "features" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[f"features.{name}"] = module
    return module


def test_price_validation_failure():
    _setup_features_stub()
    _stub_utils()
    called = {}
    _stub_expectations(called)

    price = _import_feature_module("price")
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020", periods=2, freq="T"),
        "Bid": [1.0, 1.1],
        "Ask": [1.2, 1.3],
    })
    with pytest.raises(ValueError):
        price.compute(df)
    assert called["suite"] == "price"
    assert "return" in called["columns"]


def test_cross_asset_validation_failure():
    _setup_features_stub()
    _stub_utils()
    sys.modules["data.graph_builder"] = types.SimpleNamespace(build_rolling_adjacency=lambda df: {})
    data_pkg = sys.modules.setdefault("data", types.ModuleType("data"))
    data_pkg.graph_builder = sys.modules["data.graph_builder"]
    called = {}
    _stub_expectations(called)

    cross_asset = _import_feature_module("cross_asset")
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020", periods=3, freq="D").tolist() * 2,
        "Symbol": ["A", "B", "A", "B", "A", "B"],
        "return": [0.1, 0.2, -0.1, 0.05, 0.0, -0.05],
    })
    with pytest.raises(ValueError):
        cross_asset.compute(df)
    assert called["suite"] == "cross_asset"
    assert any(c.startswith("rel_strength_") for c in called["columns"])


def test_news_validation_failure():
    _setup_features_stub()
    _stub_utils()
    _stub_data_features()
    called = {}
    _stub_expectations(called)

    news = _import_feature_module("news")
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020", periods=1, freq="T"),
        "news_summary": ["test"],
    })
    with pytest.raises(ValueError):
        news.compute(df)
    assert called["suite"] == "news"
    assert "news_sentiment" in called["columns"]
