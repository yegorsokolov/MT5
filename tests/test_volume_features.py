import numpy as np
import pandas as pd
import pytest
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "volume", Path(__file__).resolve().parents[1] / "features" / "volume.py"
)
volume = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(volume)

from strategies.baseline import BaselineStrategy


def test_volume_indicator_computation():
    data = {
        "Close": [1, 2, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.2, 2.1, 2.0, 2.1],
        "High":  [1, 2, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.2, 2.1, 2.0, 2.1],
        "Low":   [1, 2, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.2, 2.1, 2.0, 2.1],
        "Volume":[100,120,80,90,110,100,130,140,150,160,170,160,150,140,130],
    }
    df = pd.DataFrame(data)
    result = volume.compute(df)

    price = df["Close"]
    direction = np.sign(price.diff().fillna(0))
    expected_obv = (direction * df["Volume"]).cumsum().tolist()
    assert result["obv"].tolist() == expected_obv

    typical = (df["High"] + df["Low"] + price) / 3
    money_flow = typical * df["Volume"]
    pos_flow = money_flow.where(typical > typical.shift(1), 0.0)
    neg_flow = money_flow.where(typical < typical.shift(1), 0.0)
    mfr = pos_flow.rolling(14).sum() / neg_flow.rolling(14).sum().replace(0, np.nan)
    expected_mfi = 100 - 100 / (1 + mfr)
    assert result["mfi"].iloc[-1] == pytest.approx(expected_mfi.iloc[-1])


def test_baseline_volume_confirmation():
    prices = [1.0, 0.9, 1.1, 1.2]
    volume_vals = [100, 100, 100, 100]
    df = pd.DataFrame({"Close": prices, "High": prices, "Low": prices, "Volume": volume_vals})
    feats = volume.compute(df)

    bad = feats.copy()
    bad["mfi"] = 20
    strat = BaselineStrategy(short_window=2, long_window=3, rsi_window=4, atr_window=1)
    signal = 0
    for row in bad.itertuples():
        signal = strat.update(row.Close, high=row.High, low=row.Low, obv=row.obv, mfi=row.mfi)
    assert signal == 0

    good = feats.copy()
    good["mfi"] = 80
    strat = BaselineStrategy(short_window=2, long_window=3, rsi_window=4, atr_window=1)
    signal = 0
    for row in good.itertuples():
        signal = strat.update(row.Close, high=row.High, low=row.Low, obv=row.obv, mfi=row.mfi)
    assert signal == 1
