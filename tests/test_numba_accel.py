import time
import numpy as np
import pandas as pd
import importlib.util
import sys
import types
from pathlib import Path

# Load numba_accel without importing heavy utils package
utils_path = Path(__file__).resolve().parents[1] / "utils"
utils_pkg = types.ModuleType("utils")
utils_pkg.__path__ = [str(utils_path)]
sys.modules.setdefault("utils", utils_pkg)
spec = importlib.util.spec_from_file_location(
    "utils.numba_accel", utils_path / "numba_accel.py"
)
numba_accel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(numba_accel)

rolling_mean = numba_accel.rolling_mean
rolling_std = numba_accel.rolling_std
atr = numba_accel.atr
rsi = numba_accel.rsi

def _pandas_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def _timeit(func, *args, repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        best = min(best, time.perf_counter() - start)
    return best


def test_numba_matches_pandas():
    data = np.random.randn(100).astype(np.float64)
    series = pd.Series(data)
    np.testing.assert_allclose(rolling_mean(data, 14), series.rolling(14).mean().values, equal_nan=True)
    np.testing.assert_allclose(rolling_std(data, 14), series.rolling(14).std().values, equal_nan=True)
    np.testing.assert_allclose(atr(data, 14), series.diff().abs().rolling(14).mean().values, equal_nan=True)
    np.testing.assert_allclose(rsi(data, 14), _pandas_rsi(series, 14).values, equal_nan=True)


def test_numba_speed_advantage():
    data = np.random.randn(100_000).astype(np.float64)
    series = pd.Series(data)
    # warm up JIT
    rolling_mean(data, 14)
    series.rolling(14).mean()
    nb_time = _timeit(rolling_mean, data, 14)
    pd_time = _timeit(lambda arr, w: series.rolling(w).mean().values, data, 14)
    assert nb_time < pd_time
