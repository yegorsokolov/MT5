import sys
from pathlib import Path
import importlib.util

import pandas as pd
import numpy as np

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Dynamically load the microprice feature to avoid heavy imports
spec = importlib.util.spec_from_file_location(
    "microprice", ROOT / "features" / "microprice.py"
)
microprice = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(microprice)

from strategies.baseline import BaselineStrategy, IndicatorBundle


def test_microprice_computation():
    df = pd.DataFrame(
        {
            "bid_px_0": [99.0, 100.0],
            "ask_px_0": [101.0, 102.0],
            "bid_sz_0": [10.0, 15.0],
            "ask_sz_0": [5.0, 20.0],
        }
    )
    out = microprice.compute(df)
    mid = (df["bid_px_0"] + df["ask_px_0"]) / 2
    expected = (
        df["ask_px_0"] * df["bid_sz_0"] + df["bid_px_0"] * df["ask_sz_0"]
    ) / (df["bid_sz_0"] + df["ask_sz_0"])
    delta = expected - mid
    assert np.allclose(out["microprice"], expected)
    assert np.allclose(out["microprice_delta"], delta)


def test_baseline_microprice_confirmation():
    prices = [1.0, 2.0, 3.0]
    good = pd.DataFrame(
        {
            "Close": prices,
            "High": prices,
            "Low": prices,
            "bid_px_0": [p - 0.01 for p in prices],
            "ask_px_0": [p + 0.01 for p in prices],
            "bid_sz_0": [10, 11, 12],
            "ask_sz_0": [5, 4, 3],
        }
    )
    feats = microprice.compute(good)
    strat = BaselineStrategy(short_window=2, long_window=3, rsi_window=3, atr_window=1)
    signal = 0
    for row in feats.itertuples():
        ind = IndicatorBundle(high=row.High, low=row.Low, microprice_delta=row.microprice_delta)
        signal = strat.update(row.Close, ind)
    assert signal == 1

    bad = pd.DataFrame(
        {
            "Close": prices,
            "High": prices,
            "Low": prices,
            "bid_px_0": [p - 0.01 for p in prices],
            "ask_px_0": [p + 0.01 for p in prices],
            "bid_sz_0": [5, 4, 3],
            "ask_sz_0": [10, 11, 12],
        }
    )
    feats_bad = microprice.compute(bad)
    strat = BaselineStrategy(short_window=2, long_window=3, rsi_window=3, atr_window=1)
    signal = 0
    for row in feats_bad.itertuples():
        ind = IndicatorBundle(high=row.High, low=row.Low, microprice_delta=row.microprice_delta)
        signal = strat.update(row.Close, ind)
    assert signal == 0

