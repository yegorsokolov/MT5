import pandas as pd
import numpy as np
import sys
import types
from pathlib import Path


def _import_features(monkeypatch):
    stub_utils = types.ModuleType("utils")
    stub_utils.__path__ = []
    stub_utils.load_config = lambda: {
        "use_feature_cache": False,
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

    class _ResourceCaps:
        def __init__(self, cpus=1, memory_gb=1, has_gpu=False, gpu_count=0):
            self.cpus = cpus
            self.memory_gb = memory_gb
            self.has_gpu = has_gpu
            self.gpu_count = gpu_count

    stub_utils.resource_monitor = types.SimpleNamespace(
        monitor=_Mon(), ResourceCapabilities=_ResourceCaps
    )
    stub_utils.data_backend = types.SimpleNamespace(get_dataframe_module=lambda: pd)
    sys.modules["utils"] = stub_utils
    sys.modules["utils.resource_monitor"] = stub_utils.resource_monitor
    sys.modules["utils.data_backend"] = stub_utils.data_backend

    sklearn_stub = types.SimpleNamespace(
        decomposition=types.SimpleNamespace(PCA=lambda *a, **k: None),
        feature_selection=types.SimpleNamespace(
            mutual_info_classif=lambda *a, **k: np.zeros(a[0].shape[1] if a else 0)
        ),
    )
    sys.modules["sklearn"] = sklearn_stub
    sys.modules["sklearn.decomposition"] = sklearn_stub.decomposition
    sys.modules["sklearn.feature_selection"] = sklearn_stub.feature_selection

    stub_events = types.ModuleType("data.events")
    stub_events.get_events = lambda *a, **k: pd.DataFrame()
    sys.modules["data.events"] = stub_events

    stub_history = types.ModuleType("data.history")
    for name in [
        "load_history_from_urls",
        "load_history_mt5",
        "load_history_config",
        "load_history",
        "load_history_parquet",
        "load_history_memmap",
        "save_history_parquet",
        "load_multiple_histories",
    ]:
        setattr(stub_history, name, lambda *a, **k: pd.DataFrame())
    sys.modules["data.history"] = stub_history

    stub_expectations = types.ModuleType("data.expectations")
    stub_expectations.validate_dataframe = lambda df, name: df
    sys.modules["data.expectations"] = stub_expectations

    stub_graph = types.ModuleType("data.graph_builder")
    stub_graph.build_correlation_graph = lambda *a, **k: None
    stub_graph.build_rolling_adjacency = lambda *a, **k: None
    sys.modules["data.graph_builder"] = stub_graph

    sys.modules["analysis.data_lineage"] = types.SimpleNamespace(log_lineage=lambda *a, **k: None)

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    sys.modules.pop("data", None)
    sys.modules.pop("data.features", None)
    from data import features as feature_mod
    return feature_mod


def test_optimize_dtypes_memory_and_values(monkeypatch):
    feature_mod = _import_features(monkeypatch)
    raw = pd.DataFrame(
        {
            "float_col": np.arange(10, dtype=np.float64) + 0.5,
            "int_col": np.arange(10, dtype=np.int64),
        }
    )
    optimized = feature_mod.optimize_dtypes(raw.copy())
    assert optimized["float_col"].dtype == np.float32
    assert optimized["int_col"].dtype == np.int32
    assert optimized.memory_usage(deep=True).sum() < raw.memory_usage(deep=True).sum()
    pd.testing.assert_frame_equal(
        raw,
        optimized.assign(
            float_col=optimized["float_col"].astype("float64"),
            int_col=optimized["int_col"].astype("int64"),
        ),
    )


def test_make_features_downcasts(monkeypatch):
    feature_mod = _import_features(monkeypatch)
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-01", periods=10, freq="min"),
            "Bid": np.linspace(1.0, 2.0, 10),
            "Ask": np.linspace(1.0001, 2.0001, 10),
        }
    )
    result = feature_mod.make_features(df)
    num_cols = result.select_dtypes(include="number").columns
    assert not any(result[c].dtype == np.float64 for c in num_cols)
    assert not any(result[c].dtype == np.int64 for c in num_cols)
    optimized_mem = result.memory_usage(deep=True).sum()
    upcast = result.copy()
    float_cols = result.select_dtypes(include="float").columns
    int_cols = result.select_dtypes(include="int").columns
    upcast[float_cols] = upcast[float_cols].astype("float64")
    upcast[int_cols] = upcast[int_cols].astype("int64")
    upcast_mem = upcast.memory_usage(deep=True).sum()
    assert optimized_mem < upcast_mem
    round_trip = feature_mod.optimize_dtypes(upcast.copy())
    pd.testing.assert_frame_equal(result, round_trip)


def test_multi_timeframe_columns_cacheable(monkeypatch, tmp_path):
    feature_mod = _import_features(monkeypatch)
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-01", periods=10, freq="min"),
            "Bid": np.linspace(1.0, 2.0, 10),
            "Ask": np.linspace(1.0001, 2.0001, 10),
        }
    )
    result = feature_mod.make_features(df)
    assert "Bid_15m_mean" in result.columns
    assert result["Bid_15m_mean"].dtype == np.float32
    cache_path = tmp_path / "feat.pkl"
    result.to_pickle(cache_path)
    loaded = pd.read_pickle(cache_path)
    assert "Bid_15m_mean" in loaded.columns
    pd.testing.assert_series_equal(result["Bid_15m_mean"], loaded["Bid_15m_mean"])
