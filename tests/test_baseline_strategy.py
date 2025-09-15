import os
import sys
import types

# Stub minimal utils module to avoid heavy dependency initialization
utils_stub = types.ModuleType("utils")

def _dummy_load_config():
    class _Cfg:
        strategy = types.SimpleNamespace(
            session_position_limits={}, default_position_limit=1
        )

    return _Cfg()

utils_stub.load_config = _dummy_load_config
sys.modules.setdefault("utils", utils_stub)

# Ensure repo root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import deque

from strategies.baseline import BaselineStrategy, IndicatorBundle
from indicators import atr as calc_atr, bollinger, rsi as calc_rsi, sma


def test_trailing_stop_exit_long():
    strategy = BaselineStrategy(
        short_window=2,
        long_window=3,
        atr_window=2,
        atr_stop_long=1.0,
        trailing_stop_pct=0.05,
        trailing_take_profit_pct=0.05,
    )
    closes = [1, 2, 3, 2.9, 2.5]
    highs = [c + 0.1 for c in closes]
    lows = [c - 0.1 for c in closes]
    signals = [
        strategy.update(c, IndicatorBundle(high=h, low=l))
        for c, h, l in zip(closes, highs, lows)
    ]
    assert signals[2] == 1  # Buy on crossover
    assert signals[-1] == -1  # Exit via trailing stop


def test_trailing_take_profit_exit():
    strategy = BaselineStrategy(
        short_window=2,
        long_window=3,
        atr_window=2,
        atr_stop_long=0.1,
        trailing_stop_pct=0.05,
        trailing_take_profit_pct=0.05,
    )
    closes = [1, 2, 3, 3.5, 3.6, 3.3]
    highs = [c + 0.1 for c in closes]
    lows = [c - 0.1 for c in closes]
    signals = [
        strategy.update(c, IndicatorBundle(high=h, low=l))
        for c, h, l in zip(closes, highs, lows)
    ]
    assert signals[2] == 1  # Buy on crossover
    assert signals[-1] == -1  # Trailing take-profit triggers exit


def test_external_indicators_match_internal():
    strat_internal = BaselineStrategy(
        short_window=2, long_window=3, rsi_window=3, atr_window=2
    )
    strat_external = BaselineStrategy(
        short_window=2, long_window=3, rsi_window=3, atr_window=2
    )

    closes = [1, 2, 3, 2, 1.5]
    highs = [c + 0.1 for c in closes]
    lows = [c - 0.1 for c in closes]

    # Containers for manual indicator calculations
    short_q = deque(maxlen=2)
    long_q = deque(maxlen=3)
    highs_q = deque(maxlen=3)
    lows_q = deque(maxlen=3)
    closes_q = deque(maxlen=3)

    signals_internal = []
    signals_external = []

    for c, h, l in zip(closes, highs, lows):
        signals_internal.append(
            strat_internal.update(c, IndicatorBundle(high=h, low=l))
        )

        short_q.append(c)
        long_q.append(c)
        highs_q.append(h)
        lows_q.append(l)
        closes_q.append(c)

        if len(long_q) >= 3 and len(closes_q) >= 3:
            short_ma = sma(short_q, 2)
            long_ma, upper, lower = bollinger(long_q, 3)
            rsi_val = calc_rsi(long_q, 3)
            atr_val = calc_atr(highs_q, lows_q, closes_q, 2)
        else:
            short_ma = long_ma = rsi_val = atr_val = upper = lower = None

        ind = IndicatorBundle(
            high=h,
            low=l,
            short_ma=short_ma,
            long_ma=long_ma,
            rsi=rsi_val,
            atr_val=atr_val,
            boll_upper=upper,
            boll_lower=lower,
        )
        signals_external.append(strat_external.update(c, ind))

    assert signals_internal == signals_external


def test_update_accepts_evolved_indicators():
    strat = BaselineStrategy(short_window=2, long_window=3, rsi_window=3, atr_window=2)
    ind = IndicatorBundle(evolved={"custom": 1.23})
    # Should run without raising and return a numeric signal
    sig = strat.update(1.0, ind)
    assert isinstance(sig, float)
