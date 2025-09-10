import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from strategies.baseline import BaselineStrategy, IndicatorBundle

spec = importlib.util.spec_from_file_location(
    "keltner_squeeze", Path(__file__).resolve().parents[1] / "features" / "keltner_squeeze.py"
)
keltner_squeeze = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(keltner_squeeze)


def test_squeeze_break_detection():
    prices = [100.0] * 20 + [105.0]
    highs = [p + 0.1 for p in prices]
    lows = [p - 0.1 for p in prices]
    df = pd.DataFrame({"Close": prices, "High": highs, "Low": lows})
    result = keltner_squeeze.compute(df)
    assert result["squeeze_break"].iloc[-1] == 1

    prices = [100.0] * 20 + [95.0]
    highs = [p + 0.1 for p in prices]
    lows = [p - 0.1 for p in prices]
    df = pd.DataFrame({"Close": prices, "High": highs, "Low": lows})
    result = keltner_squeeze.compute(df)
    assert result["squeeze_break"].iloc[-1] == -1


def test_baseline_integration_with_squeeze():
    strategy = BaselineStrategy(short_window=2, long_window=3, atr_window=2)
    closes = [1, 2, 3]
    highs = [c + 0.1 for c in closes]
    lows = [c - 0.1 for c in closes]
    signals = []
    for c, h, l in zip(closes, highs, lows):
        ind = IndicatorBundle(high=h, low=l, squeeze_break=1)
        signals.append(strategy.update(c, ind))
    assert signals[-1] == 1

    strategy = BaselineStrategy(short_window=2, long_window=3, atr_window=2)
    signals = []
    for c, h, l in zip(closes, highs, lows):
        ind = IndicatorBundle(high=h, low=l, squeeze_break=-1)
        signals.append(strategy.update(c, ind))
    assert signals[-1] == 0
