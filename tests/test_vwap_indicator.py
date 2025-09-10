import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Load vwap feature module directly
spec = importlib.util.spec_from_file_location(
    "vwap", Path(__file__).resolve().parents[1] / "features" / "vwap.py"
)
vwap = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(vwap)

from strategies.baseline import BaselineStrategy, IndicatorBundle


def _sample_df():
    data = {
        "Timestamp": [
            "2021-03-26T00:30:00Z",
            "2021-03-26T01:00:00Z",
            "2021-03-26T09:00:00Z",
            "2021-03-26T10:00:00Z",
        ],
        "Close": [100.0, 102.0, 110.0, 112.0],
        "High": [100.0, 102.0, 110.0, 112.0],
        "Low": [100.0, 102.0, 110.0, 112.0],
        "Volume": [50, 50, 100, 100],
    }
    return pd.DataFrame(data)


def test_vwap_feature_computation():
    df = _sample_df()
    result = vwap.compute(df)
    expected_session = [100.0, 101.0, 110.0, 111.0]
    expected_day = [100.0, 101.0, 105.5, 107.6666667]
    expected_cross = [0, 0, 1, 1]
    assert np.allclose(result["vwap_session"], expected_session)
    assert np.allclose(result["vwap_day"], expected_day)
    assert result["vwap_cross"].tolist() == expected_cross


def test_baseline_vwap_gating():
    df = _sample_df().iloc[:3]
    feats = vwap.compute(df)

    strat = BaselineStrategy(short_window=2, long_window=3, rsi_window=4, atr_window=1)
    signal = 0
    for row in feats.itertuples():
        ind = IndicatorBundle(high=row.High, low=row.Low, vwap_cross=row.vwap_cross)
        signal = strat.update(row.Close, ind)
    assert signal == 1

    bad = feats.copy()
    bad["vwap_cross"] = -1
    strat = BaselineStrategy(short_window=2, long_window=3, rsi_window=4, atr_window=1)
    signal = 0
    for row in bad.itertuples():
        ind = IndicatorBundle(high=row.High, low=row.Low, vwap_cross=row.vwap_cross)
        signal = strat.update(row.Close, ind)
    assert signal == 0
