import sys
from pathlib import Path
import importlib.util

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# load divergence module directly
spec = importlib.util.spec_from_file_location(
    "divergence", ROOT / "features" / "divergence.py"
)
divergence = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(divergence)

from strategies.baseline import BaselineStrategy, IndicatorBundle


def _sample_df():
    data = {
        "Close": [100.0, 102.0, 101.0],
        "High": [100.0, 102.0, 101.0],
        "Low": [100.0, 102.0, 101.0],
        "rsi": [40.0, 30.0, 40.0],
        "macd": [0.1, 0.05, 0.1],
    }
    return pd.DataFrame(data)


def test_divergence_feature_computation():
    df = _sample_df()
    out = divergence.compute(df)
    assert out["div_rsi"].tolist() == [0, -1, 1]
    assert out["div_macd"].tolist() == [0, -1, 1]


def test_baseline_divergence_gating():
    df = divergence.compute(_sample_df())
    strat = BaselineStrategy(short_window=2, long_window=3, rsi_window=2, atr_window=1)
    signal = 0
    for row in df.itertuples():
        ind = IndicatorBundle(
            high=row.High,
            low=row.Low,
            div_rsi=row.div_rsi,
            div_macd=row.div_macd,
        )
        signal = strat.update(row.Close, ind)
    assert signal == 1

    bad = df.copy()
    bad.loc[bad.index[-1], "div_rsi"] = -1
    strat = BaselineStrategy(short_window=2, long_window=3, rsi_window=2, atr_window=1)
    signal = 0
    for row in bad.itertuples():
        ind = IndicatorBundle(
            high=row.High,
            low=row.Low,
            div_rsi=row.div_rsi,
            div_macd=row.div_macd,
        )
        signal = strat.update(row.Close, ind)
    assert signal == 0
