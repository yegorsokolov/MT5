import timeit
from pathlib import Path
import importlib.util
import sys

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


def loop_version(prices, pt_mult, sl_mult, max_horizon):
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


def test_vectorized_matches_loop_and_faster():
    np.random.seed(0)
    prices = pd.Series(np.cumprod(1 + np.random.randn(5000) * 0.001))
    pt, sl, horizon = 0.01, 0.01, 50

    ref = loop_version(prices, pt, sl, horizon)
    vec = triple_barrier(prices, pt, sl, horizon)
    pdt.assert_series_equal(vec, ref)

    loop_time = timeit.timeit(lambda: loop_version(prices, pt, sl, horizon), number=1)
    vec_time = timeit.timeit(lambda: triple_barrier(prices, pt, sl, horizon), number=1)
    assert vec_time < loop_time
