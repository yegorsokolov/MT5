import sys
import types

import numpy as np
import pandas as pd
import pytest
import importlib.util
from pathlib import Path

# Load modules directly from file paths to avoid package issues
utils_path = Path(__file__).resolve().parents[1] / "utils"

# Ensure 'utils' package exists for relative imports
utils_pkg = types.ModuleType("utils")
utils_pkg.__path__ = [str(utils_path)]
sys.modules.setdefault("utils", utils_pkg)

data_path = Path(__file__).resolve().parents[1] / "data"
data_pkg = types.ModuleType("data")
data_pkg.__path__ = [str(data_path)]
sys.modules.setdefault("data", data_pkg)

# Minimal sklearn stub for features import
sklearn_stub = types.ModuleType("sklearn")
sk_decomp = types.ModuleType("sklearn.decomposition")

class _PCA:
    def __init__(self, *args, **kwargs):
        pass

    def fit_transform(self, X):
        return X

sk_decomp.PCA = _PCA
sklearn_stub.decomposition = sk_decomp
sys.modules.setdefault("sklearn", sklearn_stub)
sys.modules.setdefault("sklearn.decomposition", sk_decomp)

# Stub requests for events module
requests_stub = types.ModuleType("requests")
sys.modules.setdefault("requests", requests_stub)

# Stub duckdb used in feature_store
duckdb_stub = types.ModuleType("duckdb")
sys.modules.setdefault("duckdb", duckdb_stub)

spec = importlib.util.spec_from_file_location(
    "utils.resource_monitor", utils_path / "resource_monitor.py"
)
rm_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm_mod)

spec = importlib.util.spec_from_file_location(
    "utils.data_backend", utils_path / "data_backend.py"
)
db_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(db_mod)
ResourceCapabilities = rm_mod.ResourceCapabilities
monitor = db_mod.monitor
get_dataframe_module = db_mod.get_dataframe_module

spec = importlib.util.spec_from_file_location(
    "data.features", Path(__file__).resolve().parents[1] / "data" / "features.py"
)
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)


def _install_stubs():
    polars_stub = types.ModuleType("polars")
    dask_stub = types.ModuleType("dask")
    dask_df_stub = types.ModuleType("dask.dataframe")
    for mod in (polars_stub, dask_df_stub):
        for attr in ["DataFrame", "Series"]:
            setattr(mod, attr, getattr(pd, attr))
    sys.modules.setdefault("polars", polars_stub)
    sys.modules.setdefault("dask", dask_stub)
    sys.modules.setdefault("dask.dataframe", dask_df_stub)


@pytest.fixture(autouse=True)
def setup_stubs():
    _install_stubs()


def test_backend_selection(monkeypatch):
    # Start with pandas
    monkeypatch.setattr(
        monitor,
        "capabilities",
        ResourceCapabilities(cpus=2, memory_gb=2, has_gpu=False, gpu_count=0),
    )
    mod = get_dataframe_module()
    assert mod.__name__ == "pandas"

    # Switch to polars
    monkeypatch.setattr(
        monitor,
        "capabilities",
        ResourceCapabilities(cpus=12, memory_gb=24, has_gpu=False, gpu_count=0),
    )
    mod = get_dataframe_module()
    assert mod.__name__ == "polars"

    # Switch to dask
    monkeypatch.setattr(
        monitor,
        "capabilities",
        ResourceCapabilities(cpus=32, memory_gb=64, has_gpu=False, gpu_count=0),
    )
    mod = get_dataframe_module()
    assert mod.__name__ == "dask.dataframe"


def test_feature_consistency_across_backends(monkeypatch):
    base_caps = ResourceCapabilities(cpus=2, memory_gb=2, has_gpu=False, gpu_count=0)
    monkeypatch.setattr(monitor, "capabilities", base_caps)
    pd_mod = get_dataframe_module()
    df = pd_mod.DataFrame({
        "close": [1, 2, 3, 4, 5],
        "ma_10": [1, 2, 3, 4, 5],
        "ma_30": [2, 2, 2, 2, 2],
    })
    rsi_base = features.compute_rsi(df["close"], period=2).fillna(0)
    sig_base = features.ma_cross_signal(df, "ma_10", "ma_30")

    # Polars backend
    monkeypatch.setattr(
        monitor,
        "capabilities",
        ResourceCapabilities(cpus=12, memory_gb=24, has_gpu=False, gpu_count=0),
    )
    pl_mod = get_dataframe_module()
    df_pl = pl_mod.DataFrame({
        "close": [1, 2, 3, 4, 5],
        "ma_10": [1, 2, 3, 4, 5],
        "ma_30": [2, 2, 2, 2, 2],
    })
    rsi_pl = features.compute_rsi(df_pl["close"], period=2).fillna(0)
    sig_pl = features.ma_cross_signal(df_pl, "ma_10", "ma_30")

    # Dask backend
    monkeypatch.setattr(
        monitor,
        "capabilities",
        ResourceCapabilities(cpus=32, memory_gb=64, has_gpu=False, gpu_count=0),
    )
    dd_mod = get_dataframe_module()
    df_dd = dd_mod.DataFrame({
        "close": [1, 2, 3, 4, 5],
        "ma_10": [1, 2, 3, 4, 5],
        "ma_30": [2, 2, 2, 2, 2],
    })
    rsi_dd = features.compute_rsi(df_dd["close"], period=2).fillna(0)
    sig_dd = features.ma_cross_signal(df_dd, "ma_10", "ma_30")

    assert np.allclose(rsi_base.values, rsi_pl.values)
    assert np.allclose(rsi_base.values, rsi_dd.values)
    assert sig_base.equals(sig_pl)
    assert sig_base.equals(sig_dd)
