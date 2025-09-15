import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

# Ensure repository root on path and import triple_barrier directly from file
sys.path.append(str(Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location(
    "labels", Path(__file__).resolve().parents[1] / "data" / "labels.py"
)
labels_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labels_mod)
triple_barrier = labels_mod.triple_barrier


def legacy_loop(prices, pt_mult, sl_mult, max_horizon):
    labels = pd.Series(0, index=prices.index, dtype=int)
    n = len(prices)
    for i in range(n):
        p0 = prices.iloc[i]
        upper = p0 * (1 + pt_mult)
        lower = p0 * (1 - sl_mult)
        end = min(i + max_horizon, n - 1)
        outcome = 0
        for j in range(i + 1, end + 1):
            p = prices.iloc[j]
            if p >= upper:
                outcome = 1
                break
            if p <= lower:
                outcome = -1
                break
        labels.iloc[i] = outcome
    return labels


def _timeit(func, *args, repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        best = min(best, time.perf_counter() - start)
    return best


def test_numba_matches_legacy_loop():
    np.random.seed(0)
    prices = pd.Series(np.cumprod(1 + np.random.randn(5000) * 0.001))
    pt, sl, horizon = 0.01, 0.01, 50

    expected = legacy_loop(prices, pt, sl, horizon)
    result = triple_barrier(prices, pt, sl, horizon)
    pdt.assert_series_equal(result, expected)


def test_numba_speed_advantage_over_loop():
    np.random.seed(1)
    prices = pd.Series(np.cumprod(1 + np.random.randn(6000) * 0.001))
    pt, sl, horizon = 0.01, 0.01, 60

    # warm-up to allow JIT compilation
    triple_barrier(prices, pt, sl, horizon)
    legacy_loop(prices, pt, sl, horizon)

    new_time = _timeit(triple_barrier, prices, pt, sl, horizon)
    legacy_time = _timeit(legacy_loop, prices, pt, sl, horizon)
    assert new_time < legacy_time
