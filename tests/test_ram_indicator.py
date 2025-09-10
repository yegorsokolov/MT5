import os
import sys
import pandas as pd
import pytest
import importlib.util
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

spec = importlib.util.spec_from_file_location(
    "ram", Path(__file__).resolve().parents[1] / "features" / "ram.py"
)
ram = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ram)

from strategies.baseline import BaselineStrategy


def test_ram_indicator_computation():
    df = pd.DataFrame({"Close": [1, 1.1, 1.2, 1.15, 1.3]})
    result = ram.compute(df, window=3)

    ret = df["Close"].pct_change()
    mean_ret = ret.rolling(3).mean()
    vol = ret.rolling(3).std()
    expected = (mean_ret / vol).fillna(0.0)
    assert result["ram"].tolist() == pytest.approx(expected.tolist())


def test_baseline_ram_gating():
    prices = [1.0, 1.1, 1.2]
    df = pd.DataFrame({"Close": prices, "High": prices, "Low": prices})
    feats = ram.compute(df, window=2)

    strat = BaselineStrategy(
        short_window=2,
        long_window=3,
        rsi_window=4,
        atr_window=1,
        ram_long_threshold=0.0,
        ram_short_threshold=0.0,
    )
    signal = 0
    for row in feats.itertuples():
        signal = strat.update(row.Close, high=row.High, low=row.Low, ram=row.ram)
    assert signal == 1

    strat = BaselineStrategy(
        short_window=2,
        long_window=3,
        rsi_window=4,
        atr_window=1,
        ram_long_threshold=20.0,
        ram_short_threshold=0.0,
    )
    signal = 0
    for row in feats.itertuples():
        signal = strat.update(row.Close, high=row.High, low=row.Low, ram=row.ram)
    assert signal == 0

    prices = [3.0, 2.0, 1.0]
    df = pd.DataFrame({"Close": prices, "High": prices, "Low": prices})
    feats = ram.compute(df, window=2)

    strat = BaselineStrategy(
        short_window=2,
        long_window=3,
        rsi_window=4,
        atr_window=1,
        ram_long_threshold=0.0,
        ram_short_threshold=0.0,
    )
    signal = 0
    for row in feats.itertuples():
        signal = strat.update(row.Close, high=row.High, low=row.Low, ram=row.ram)
    assert signal == -1

    strat = BaselineStrategy(
        short_window=2,
        long_window=3,
        rsi_window=4,
        atr_window=1,
        ram_long_threshold=0.0,
        ram_short_threshold=-5.0,
    )
    signal = 0
    for row in feats.itertuples():
        signal = strat.update(row.Close, high=row.High, low=row.Low, ram=row.ram)
    assert signal == 0
