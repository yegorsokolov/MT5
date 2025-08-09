import pandas as pd
import numpy as np
import sys
import types
import time
from pathlib import Path


def test_make_features_uses_cache(monkeypatch, tmp_path):
    stub_utils = types.ModuleType("utils")
    stub_utils.load_config = lambda: {
        "use_feature_cache": True,
        "use_atr": False,
        "use_donchian": False,
        "use_dask": False,
    }

    class _Mon:
        def start(self):
            pass

        class capabilities:
            @staticmethod
            def model_size():
                return "small"

    stub_utils.resource_monitor = types.SimpleNamespace(monitor=_Mon())
    sys.modules["utils"] = stub_utils
    sys.modules["utils.resource_monitor"] = stub_utils.resource_monitor
    sklearn_stub = types.SimpleNamespace(
        decomposition=types.SimpleNamespace(PCA=lambda *a, **k: None)
    )
    sys.modules["sklearn"] = sklearn_stub
    sys.modules["sklearn.decomposition"] = sklearn_stub.decomposition

    sys.path.append(str(Path(__file__).resolve().parents[1]))
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
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True),
        second.reset_index(drop=True),
        check_dtype=False,
    )

