import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Load macd feature module directly
spec = importlib.util.spec_from_file_location(
    "macd", ROOT / "features" / "macd.py"
)
macd = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(macd)

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
    }
    return pd.DataFrame(data)


def test_macd_feature_computation():
    df = _sample_df()
    result = macd.compute(df)
    expected_macd = [0.0, 0.15954415954415424, 0.9209016160582877, 1.6664563000388029]
    expected_signal = [0.0, 0.03190883190883085, 0.20970738873872224, 0.5010571709987384]
    expected_cross = [0, 1, 1, 1]
    assert np.allclose(result["macd"], expected_macd)
    assert np.allclose(result["macd_signal"], expected_signal)
    assert result["macd_cross"].tolist() == expected_cross


def test_baseline_macd_gating():
    df = _sample_df().iloc[:3]
    feats = macd.compute(df)

    strat = BaselineStrategy(short_window=2, long_window=3, rsi_window=4, atr_window=1)
    signal = 0
    for row in feats.itertuples():
        ind = IndicatorBundle(high=row.High, low=row.Low, macd_cross=row.macd_cross)
        signal = strat.update(row.Close, ind)
    assert signal == 1

    bad = feats.copy()
    bad["macd_cross"] = -1
    strat = BaselineStrategy(short_window=2, long_window=3, rsi_window=4, atr_window=1)
    signal = 0
    for row in bad.itertuples():
        ind = IndicatorBundle(high=row.High, low=row.Low, macd_cross=row.macd_cross)
        signal = strat.update(row.Close, ind)
    assert signal == 0
