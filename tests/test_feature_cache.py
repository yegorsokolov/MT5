import pandas as pd
import numpy as np
import sys
import types
import time
from pathlib import Path


def test_make_features_uses_cache(monkeypatch, tmp_path):
    stub_utils = types.ModuleType("utils")
    stub_utils.__path__ = []
    stub_utils.load_config = lambda: {
        "use_feature_cache": True,
        "use_atr": False,
        "use_donchian": False,
        "use_dask": False,
        "multi_timeframes": ["15min", "1H"],
    }

    class _Mon:
        def start(self):
            pass
        capability_tier = "lite"

        class capabilities:
            cpus = 1
            memory_gb = 1
            has_gpu = False
            @staticmethod
            def capability_tier():
                return "lite"

    stub_utils.resource_monitor = types.SimpleNamespace(monitor=_Mon())
    stub_utils.data_backend = types.SimpleNamespace(get_dataframe_module=lambda: pd)
    sys.modules["utils"] = stub_utils
    sys.modules["utils.resource_monitor"] = stub_utils.resource_monitor
    sys.modules["utils.data_backend"] = stub_utils.data_backend
    sklearn_stub = types.SimpleNamespace(
        decomposition=types.SimpleNamespace(PCA=lambda *a, **k: None)
    )
    sys.modules["sklearn"] = sklearn_stub
    sys.modules["sklearn.decomposition"] = sklearn_stub.decomposition

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    sys.modules.pop("data", None)
    sys.modules.pop("data.features", None)
    from data import features as feature_mod
    from data import feature_store as fs_mod

    # patch FeatureStore to use a temporary cache file
    monkeypatch.setattr(
        feature_mod,
        "FeatureStore",
        lambda: fs_mod.FeatureStore(tmp_path / "feature_store.duckdb"),
    )

    # slow down compute_rsi so cache hit shows clear runtime reduction
    original_rsi = feature_mod.compute_rsi

    def slow_rsi(series, period):  # pragma: no cover - timing helper
        time.sleep(0.05)
        return original_rsi(series, period)

    monkeypatch.setattr(feature_mod, "compute_rsi", slow_rsi)

    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-01", periods=60, freq="min"),
            "Bid": np.linspace(1.0, 2.0, 60),
            "Ask": np.linspace(1.0001, 2.0001, 60),
        }
    )

    t0 = time.time()
    first = feature_mod.make_features(df)
    first_time = time.time() - t0
    cache_file = tmp_path / "feature_store.duckdb"
    assert cache_file.exists()

    # multi-timeframe features should be present
    assert "mid_15m_mean" in first.columns
    assert "mid_1h_mean" in first.columns

    # numeric columns should be downcast from 64-bit types
    num_cols = first.select_dtypes(include="number").columns
    assert not any(first[c].dtype == np.float64 for c in num_cols)
    assert not any(first[c].dtype == np.int64 for c in num_cols)

    # optimized DataFrame should use less memory than the upcast equivalent
    optimized_mem = first.memory_usage(deep=True).sum()
    upcast = first.copy()
    float_cols = first.select_dtypes(include="float").columns
    int_cols = first.select_dtypes(include="int").columns
    upcast[float_cols] = upcast[float_cols].astype("float64")
    upcast[int_cols] = upcast[int_cols].astype("int64")
    upcast_mem = upcast.memory_usage(deep=True).sum()
    assert optimized_mem < upcast_mem

    logs = []
    monkeypatch.setattr(
        feature_mod.logger,
        "info",
        lambda msg, *a: logs.append(msg % a if a else msg),
    )
    t1 = time.time()
    second = feature_mod.make_features(df)
    second_time = time.time() - t1
    assert second_time < first_time
    assert any("Loading features from cache" in m for m in logs)
    assert "mid_15m_mean" in second.columns
    assert "mid_1h_mean" in second.columns
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True),
        second.reset_index(drop=True),
        check_dtype=False,
    )


def test_cross_asset_features_cached(monkeypatch, tmp_path):
    stub_utils = types.ModuleType("utils")
    stub_utils.__path__ = []
    stub_utils.load_config = lambda: {
        "use_feature_cache": True,
        "use_atr": False,
        "use_donchian": False,
        "use_dask": False,
    }

    class _Mon:
        def start(self):
            pass
        capability_tier = "lite"

        class capabilities:
            cpus = 1
            memory_gb = 1
            has_gpu = False
            @staticmethod
            def capability_tier():
                return "lite"

    stub_utils.resource_monitor = types.SimpleNamespace(monitor=_Mon())
    stub_utils.data_backend = types.SimpleNamespace(get_dataframe_module=lambda: pd)
    sys.modules["utils"] = stub_utils
    sys.modules["utils.resource_monitor"] = stub_utils.resource_monitor
    sys.modules["utils.data_backend"] = stub_utils.data_backend

    class _PCA:
        def __init__(self, *a, **k):
            self.n_components = k.get("n_components", 1)

        def fit_transform(self, X):  # pragma: no cover - simple stub
            return np.zeros((len(X), self.n_components))

    sklearn_stub = types.SimpleNamespace(
        decomposition=types.SimpleNamespace(PCA=_PCA)
    )
    sys.modules["sklearn"] = sklearn_stub
    sys.modules["sklearn.decomposition"] = sklearn_stub.decomposition

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    sys.modules.pop("data", None)
    sys.modules.pop("data.features", None)
    from data import features as feature_mod
    from data import feature_store as fs_mod

    monkeypatch.setattr(
        feature_mod,
        "FeatureStore",
        lambda: fs_mod.FeatureStore(tmp_path / "feature_store.duckdb"),
    )

    ts = pd.date_range("2020-01-01", periods=60, freq="min")
    df_a = pd.DataFrame(
        {
            "Timestamp": ts,
            "Bid": np.linspace(1.0, 1.5, 60),
            "Ask": np.linspace(1.0001, 1.5001, 60),
            "Symbol": "AAA",
        }
    )
    df_b = pd.DataFrame(
        {
            "Timestamp": ts,
            "Bid": np.linspace(2.0, 2.5, 60),
            "Ask": np.linspace(2.0001, 2.5001, 60),
            "Symbol": "BBB",
        }
    )
    df = pd.concat([df_a, df_b], ignore_index=True)

    first = feature_mod.make_features(df)
    assert any(col.startswith("AAA_BBB_corr_30") for col in first.columns)

    second = feature_mod.make_features(df)
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True),
        second.reset_index(drop=True),
        check_dtype=False,
    )

