import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "supertrend", Path(__file__).resolve().parents[1] / "features" / "supertrend.py"
)
supertrend = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(supertrend)


def reference_supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(period).mean()
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    st = pd.Series(np.nan, index=df.index)
    direction = 0
    for i in range(period, len(df)):
        if direction <= 0 and close.iloc[i] > upper.iloc[i - 1]:
            direction = 1
        elif direction >= 0 and close.iloc[i] < lower.iloc[i - 1]:
            direction = -1
        st.iloc[i] = lower.iloc[i] if direction == 1 else upper.iloc[i]
    return st


def test_supertrend_indicator():
    close = [
        10.0,
        10.5,
        10.2,
        10.8,
        11.0,
        10.7,
        10.9,
        11.2,
        11.5,
        11.3,
        11.6,
        11.8,
        11.4,
        11.2,
        11.5,
    ]
    data = {
        "Close": close,
        "High": [c + 0.1 for c in close],
        "Low": [c - 0.1 for c in close],
    }
    df = pd.DataFrame(data)
    result = supertrend.compute(df)
    expected = reference_supertrend(df)
    assert result["supertrend"].iloc[-1] == pytest.approx(expected.iloc[-1])

    price = df["Close"]
    expected_break = (
        (
            (
                (price > result["supertrend"])
                & (price.shift(1) <= result["supertrend"].shift(1))
            ).astype(int)
            - (
                (price < result["supertrend"])
                & (price.shift(1) >= result["supertrend"].shift(1))
            ).astype(int)
        )
        .fillna(0)
        .astype(int)
    )
    assert result["supertrend_break"].tolist() == expected_break.tolist()
