import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "kalman_ma", Path(__file__).resolve().parents[1] / "features" / "kalman_ma.py"
)
kalman_ma = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(kalman_ma)


def reference_kma(series: pd.Series, q: float, r: float) -> pd.Series:
    x = float(series.iloc[0])
    p = 1.0
    estimates = [x]
    for z in series.iloc[1:]:
        p = p + q
        k = p / (p + r)
        x = x + k * (float(z) - x)
        p = (1 - k) * p
        estimates.append(x)
    return pd.Series(estimates, index=series.index)


def test_kma_shape_and_cross():
    close = [1.0] * 5 + [2.0] * 5
    df = pd.DataFrame({"Close": close})
    result = kalman_ma.compute(df, process_noise=1e-5, measurement_noise=1e-2)
    expected = reference_kma(df["Close"], 1e-5, 1e-2)
    assert result["kma"].equals(expected)
    assert len(result) == len(df)
    price = df["Close"]
    expected_cross = (
        ((price > expected) & (price.shift(1) <= expected.shift(1))).astype(int)
        - ((price < expected) & (price.shift(1) >= expected.shift(1))).astype(int)
    ).fillna(0).astype(int)
    assert result["kma_cross"].tolist() == expected_cross.tolist()
    # Filter responds to the step but retains some lag
    assert 1.5 < result["kma"].iloc[-1] < 2.0
