import os
import sys
from pathlib import Path
import importlib.util

import pandas as pd
import numpy as np

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Load order_flow module directly to avoid heavy package imports
spec = importlib.util.spec_from_file_location(
    "order_flow", ROOT / "features" / "order_flow.py"
)
order_flow = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(order_flow)

from strategies.baseline import BaselineStrategy, IndicatorBundle


def test_order_flow_feature_computation():
    df = pd.DataFrame(
        {
            "bid_sz_0": [10.0, 12.0, 8.0],
            "ask_sz_0": [5.0, 4.0, 7.0],
        }
    )
    out = order_flow.compute(df, window=2)
    expected_delta = df["bid_sz_0"] - df["ask_sz_0"]
    expected_imb = expected_delta / (df["bid_sz_0"] + df["ask_sz_0"])
    expected_cvd = expected_delta.cumsum()
    assert np.allclose(out["imbalance"], expected_imb)
    assert np.allclose(out["cvd"], expected_cvd)
    assert np.allclose(
        out["cvd_roll_mean"], expected_cvd.rolling(2).mean(), equal_nan=True
    )
    assert np.allclose(
        out["imbalance_roll_mean"], expected_imb.rolling(2).mean(), equal_nan=True
    )


def test_baseline_cvd_confirmation():
    prices = [1.0, 2.0, 3.0]
    good = pd.DataFrame(
        {
            "Close": prices,
            "High": prices,
            "Low": prices,
            "bid_sz_0": [10, 11, 12],
            "ask_sz_0": [8, 7, 6],
        }
    )
    feats = order_flow.compute(good)
    strat = BaselineStrategy(short_window=2, long_window=3, rsi_window=3, atr_window=1)
    signal = 0
    for row in feats.itertuples():
        ind = IndicatorBundle(high=row.High, low=row.Low, cvd=row.cvd)
        signal = strat.update(row.Close, ind)
    assert signal == 1

    bad = pd.DataFrame(
        {
            "Close": prices,
            "High": prices,
            "Low": prices,
            "bid_sz_0": [10, 9, 8],
            "ask_sz_0": [8, 11, 12],
        }
    )
    feats_bad = order_flow.compute(bad)
    strat = BaselineStrategy(short_window=2, long_window=3, rsi_window=3, atr_window=1)
    signal = 0
    for row in feats_bad.itertuples():
        ind = IndicatorBundle(high=row.High, low=row.Low, cvd=row.cvd)
        signal = strat.update(row.Close, ind)
    assert signal == 0
