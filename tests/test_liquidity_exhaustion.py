import pandas as pd
import runpy
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from strategies.baseline import BaselineStrategy, IndicatorBundle

compute = runpy.run_path("features/liquidity_exhaustion.py")["compute"]


def _sample_df():
    return pd.DataFrame(
        {
            "bid_px_0": [99, 99],
            "bid_px_1": [98, 98],
            "ask_px_0": [101, 101],
            "ask_px_1": [102, 102],
            "bid_sz_0": [10, 1],
            "bid_sz_1": [10, 1],
            "ask_sz_0": [1, 10],
            "ask_sz_1": [1, 10],
        }
    )


def test_compute_liquidity_exhaustion():
    df = _sample_df()
    out = compute(df, ticks=1, upper_ratio=2.0, lower_ratio=0.5)
    assert list(out["liq_ratio"].round(1)) == [10.0, 0.1]
    assert list(out["liq_exhaustion"]) == [1, -1]


def test_baseline_filter():
    strat = BaselineStrategy(short_window=1, long_window=2, rsi_window=1, atr_window=1)
    # prime internal state
    strat.update(price=1.0)
    ind = IndicatorBundle(
        short_ma=2.0,
        long_ma=1.0,
        rsi=50.0,
        boll_upper=10.0,
        boll_lower=0.0,
        liq_exhaustion=-1,
    )
    sig = strat.update(price=2.0, indicators=ind)
    assert sig == 0
