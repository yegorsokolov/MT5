import pandas as pd
import sys
import types
from pathlib import Path


def _setup_feature_env(monkeypatch, tmp_path):
    """Prepare stub modules so `make_features` can run in isolation."""
    stub_utils = types.ModuleType("utils")
    stub_utils.__path__ = []
    stub_utils.load_config = lambda: {
        "use_feature_cache": True,
        "use_atr": False,
        "use_donchian": False,
        "use_kalman": False,
        "use_dask": False,
    }

    class _Mon:
        def start(self):
            pass
        capability_tier = "lite"

        class capabilities:
            cpus = 1

            @staticmethod
            def capability_tier():
                return "lite"

    stub_utils.resource_monitor = types.SimpleNamespace(monitor=_Mon())
    stub_utils.data_backend = types.SimpleNamespace(get_dataframe_module=lambda: pd)
    sys.modules["utils"] = stub_utils
    sys.modules["utils.resource_monitor"] = stub_utils.resource_monitor
    sys.modules["utils.data_backend"] = stub_utils.data_backend
    sys.modules.setdefault("requests", types.SimpleNamespace(get=lambda *a, **k: None))

    sklearn_stub = types.SimpleNamespace(
        decomposition=types.SimpleNamespace(PCA=lambda *a, **k: None)
    )
    sys.modules["sklearn"] = sklearn_stub
    sys.modules["sklearn.decomposition"] = sklearn_stub.decomposition

    root = Path(__file__).resolve().parents[1]
    sys.path.append(str(root))
    for mod in [
        "data",
        "data.features",
        "data.history",
        "data.feature_store",
        "data.versioning",
        "data.delta_store",
    ]:
        sys.modules.pop(mod, None)

    pkg = types.ModuleType("data")
    pkg.__path__ = [str(root / "data")]
    sys.modules["data"] = pkg

    import importlib.util

    def _load(name):
        spec = importlib.util.spec_from_file_location(
            name, root / (name.replace(".", "/") + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "data"
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        sys.modules[name] = module
        return module

    fs_mod = _load("data.feature_store")
    feature_mod = _load("data.features")
    history_mod = _load("data.history")

    # ensure FeatureStore uses temporary path
    monkeypatch.setattr(
        feature_mod,
        "FeatureStore",
        lambda: fs_mod.FeatureStore(tmp_path / "feature_store.duckdb"),
    )
    return feature_mod, history_mod


def test_delta_processing(monkeypatch, tmp_path):
    feature_mod, history_mod = _setup_feature_env(monkeypatch, tmp_path)

    csv_path = tmp_path / "EURUSD_history.csv"
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-01", periods=5, freq="min"),
            "Bid": [1.0, 1.1, 1.2, 1.3, 1.4],
            "Ask": [1.0002, 1.1002, 1.2002, 1.3002, 1.4002],
        }
    )
    df.to_csv(csv_path, index=False, date_format="%Y%m%d %H:%M:%S:%f")

    first_hist = history_mod.load_history(csv_path)
    first_feat = feature_mod.make_features(first_hist)

    delta_file = csv_path.with_suffix(".csv.delta")
    assert not delta_file.exists()

    # append new records
    new_rows = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-01 00:05", periods=3, freq="min"),
            "Bid": [1.5, 1.6, 1.7],
            "Ask": [1.5002, 1.6002, 1.7002],
        }
    )
    new_rows.to_csv(
        csv_path, mode="a", header=False, index=False, date_format="%Y%m%d %H:%M:%S:%f"
    )

    second_hist = history_mod.load_history(csv_path)
    second_feat = feature_mod.make_features(second_hist)

    assert delta_file.exists()
    delta_df = pd.read_csv(delta_file)
    assert len(delta_df) == len(new_rows)

    # original features should remain intact
    pd.testing.assert_frame_equal(
        first_feat.reset_index(drop=True),
        second_feat.iloc[: len(first_feat)].reset_index(drop=True),
        check_dtype=False,
    )
    assert len(second_feat) == len(first_feat) + len(new_rows)

    # a third run without new data should not change anything
    third_hist = history_mod.load_history(csv_path)
    third_feat = feature_mod.make_features(third_hist)
    pd.testing.assert_frame_equal(second_feat, third_feat, check_dtype=False)
    delta_df2 = pd.read_csv(delta_file)
    assert len(delta_df2) == len(delta_df)
