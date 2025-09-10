import os
import sys

# Ensure repo root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from strategies.baseline import BaselineStrategy


def test_baseline_hurst_gating():
    prices = [1.0, 1.1, 1.2]
    strat = BaselineStrategy(
        short_window=2,
        long_window=3,
        rsi_window=4,
        atr_window=1,
        hurst_trend_min=0.5,
        hurst_mean_reversion_max=0.5,
    )
    signal = 0
    for p in prices:
        signal = strat.update(p, hurst=0.8)
    assert signal == 1

    strat = BaselineStrategy(
        short_window=2,
        long_window=3,
        rsi_window=4,
        atr_window=1,
        hurst_trend_min=0.5,
        hurst_mean_reversion_max=0.5,
    )
    signal = 0
    for p in prices:
        signal = strat.update(p, hurst=0.3)
    assert signal == 0

    prices = [3.0, 2.0, 1.0]
    strat = BaselineStrategy(
        short_window=2,
        long_window=3,
        rsi_window=4,
        atr_window=1,
        hurst_trend_min=0.5,
        hurst_mean_reversion_max=0.5,
    )
    signal = 0
    for p in prices:
        signal = strat.update(p, hurst=0.3)
    assert signal == -1

    strat = BaselineStrategy(
        short_window=2,
        long_window=3,
        rsi_window=4,
        atr_window=1,
        hurst_trend_min=0.5,
        hurst_mean_reversion_max=0.5,
    )
    signal = 0
    for p in prices:
        signal = strat.update(p, hurst=0.8)
    assert signal == 0
