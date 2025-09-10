import os
import sys

# Ensure repo root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from strategies.baseline import BaselineStrategy, IndicatorBundle


def test_compute_signal_basic_cross():
    strat = BaselineStrategy(short_window=2, long_window=3, rsi_window=3, atr_window=1)
    prices = [1.0, 2.0, 3.0]
    signals = []
    for p in prices:
        signals.append(strat._compute_signal(p, IndicatorBundle(high=p, low=p)))
    assert signals[-1] == 1


def test_apply_filters_obv_mfi_block():
    strat = BaselineStrategy(short_window=2, long_window=3, rsi_window=3, atr_window=1)
    # prime with some data
    for p in [1.0, 2.0, 3.0]:
        strat._compute_signal(p, IndicatorBundle(high=p, low=p))
    raw = 1
    filt = strat._apply_filters(raw, IndicatorBundle(obv=1.0, mfi=20.0), 3.0)
    assert filt == 0


def test_manage_position_entry_and_exit():
    strat = BaselineStrategy(short_window=2, long_window=3, rsi_window=3, atr_window=1, long_regimes={1})
    for p in [1.0, 2.0, 3.0]:
        strat._compute_signal(p, IndicatorBundle(high=p, low=p))
    sig = strat._manage_position(3.0, 1, regime=1)
    assert sig == 1 and strat.position == 1
    exit_sig = strat._manage_position(4.0, 0, regime=0)
    assert exit_sig == -1 and strat.position == 0
