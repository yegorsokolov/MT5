import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "adaptive_ma", Path(__file__).resolve().parents[1] / "features" / "adaptive_ma.py"
)
adaptive_ma = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(adaptive_ma)


def reference_kama(
    series: pd.Series, period: int = 10, fast: int = 2, slow: int = 30
) -> pd.Series:
    change = series.diff(period).abs()
    volatility = series.diff().abs().rolling(period).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    kama = series.copy()
    for i in range(period, len(series)):
        kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (
            series.iloc[i] - kama.iloc[i - 1]
        )
    return kama


def test_kama_indicator():
    close = [
        1.0,
        1.05,
        1.1,
        1.15,
        1.2,
        1.25,
        1.3,
        1.35,
        1.4,
        1.45,
        1.5,
        1.48,
        1.46,
        1.44,
        1.42,
        1.4,
        1.38,
        1.36,
        1.34,
        1.32,
    ]
    df = pd.DataFrame({"Close": close})
    result = adaptive_ma.compute(df)
    expected = reference_kama(df["Close"])
    assert result["kama"].iloc[-1] == pytest.approx(expected.iloc[-1])

    price = df["Close"]
    expected_cross = (
        (
            (
                (price > result["kama"]) & (price.shift(1) <= result["kama"].shift(1))
            ).astype(int)
            - (
                (price < result["kama"]) & (price.shift(1) >= result["kama"].shift(1))
            ).astype(int)
        )
        .fillna(0)
        .astype(int)
    )
    assert result["kama_cross"].tolist() == expected_cross.tolist()
