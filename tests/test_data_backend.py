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

# Stub analytics.metrics_store used in resource_monitor
analytics_stub = types.ModuleType("analytics")
metrics_store_stub = types.ModuleType("analytics.metrics_store")

def _record_metric(name, value):
    return None

metrics_store_stub.record_metric = _record_metric
analytics_stub.metrics_store = metrics_store_stub
sys.modules.setdefault("analytics", analytics_stub)
sys.modules.setdefault("analytics.metrics_store", metrics_store_stub)

# Stub analysis modules used in features
analysis_stub = types.ModuleType("analysis")
garch_stub = types.ModuleType("analysis.garch_vol")

def garch_volatility(*args, **kwargs):
    return pd.Series([0])

garch_stub.garch_volatility = garch_volatility
kalman_stub = types.ModuleType("analysis.kalman_filter")

def kalman_smooth(*args, **kwargs):
    return pd.Series([0])

kalman_stub.kalman_smooth = kalman_smooth
freq_stub = types.ModuleType("analysis.frequency_features")

def rolling_fft_features(*args, **kwargs):
    return pd.DataFrame()

def rolling_wavelet_features(*args, **kwargs):
    return pd.DataFrame()

freq_stub.rolling_fft_features = rolling_fft_features
freq_stub.rolling_wavelet_features = rolling_wavelet_features
session_stub = types.ModuleType("analysis.session_features")

def add_session_features(df):
    return df

session_stub.add_session_features = add_session_features
analysis_stub.garch_vol = garch_stub
analysis_stub.kalman_filter = kalman_stub
analysis_stub.frequency_features = freq_stub
analysis_stub.session_features = session_stub
sys.modules.setdefault("analysis", analysis_stub)
sys.modules.setdefault("analysis.garch_vol", garch_stub)
sys.modules.setdefault("analysis.kalman_filter", kalman_stub)
sys.modules.setdefault("analysis.frequency_features", freq_stub)
sys.modules.setdefault("analysis.session_features", session_stub)

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
    cudf_stub = types.ModuleType("cudf")
    for mod in (polars_stub, dask_df_stub, cudf_stub):
        for attr in [
            "DataFrame",
            "Series",
            "concat",
            "read_csv",
            "read_parquet",
            "merge_asof",
            "to_numeric",
            "to_datetime",
        ]:
            if hasattr(pd, attr):
                setattr(mod, attr, getattr(pd, attr))
    sys.modules.setdefault("polars", polars_stub)
    sys.modules.setdefault("dask", dask_stub)
    sys.modules.setdefault("dask.dataframe", dask_df_stub)
    sys.modules.setdefault("cudf", cudf_stub)


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

    # Switch to cuDF when GPU is available
    monkeypatch.setattr(
        monitor,
        "capabilities",
        ResourceCapabilities(cpus=8, memory_gb=32, has_gpu=True, gpu_count=1),
    )
    mod = get_dataframe_module()
    assert mod.__name__ == "cudf"


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

    # cuDF backend
    monkeypatch.setattr(
        monitor,
        "capabilities",
        ResourceCapabilities(cpus=8, memory_gb=32, has_gpu=True, gpu_count=1),
    )
    cd_mod = get_dataframe_module()
    df_cd = cd_mod.DataFrame({
        "close": [1, 2, 3, 4, 5],
        "ma_10": [1, 2, 3, 4, 5],
        "ma_30": [2, 2, 2, 2, 2],
    })
    rsi_cd = features.compute_rsi(df_cd["close"], period=2).fillna(0)
    sig_cd = features.ma_cross_signal(df_cd, "ma_10", "ma_30")
    rsi_cd_vals = (
        rsi_cd.to_pandas().values if hasattr(rsi_cd, "to_pandas") else rsi_cd.values
    )
    sig_cd_df = sig_cd.to_pandas() if hasattr(sig_cd, "to_pandas") else sig_cd

    assert np.allclose(rsi_base.values, rsi_pl.values)
    assert np.allclose(rsi_base.values, rsi_dd.values)
    assert np.allclose(rsi_base.values, rsi_cd_vals)
    assert sig_base.equals(sig_pl)
    assert sig_base.equals(sig_dd)
    assert sig_base.equals(sig_cd_df)
