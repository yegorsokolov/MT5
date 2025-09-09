import os
import sys

# Ensure repo root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from strategies.baseline import BaselineStrategy


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
    signals = [strategy.update(c, h, l) for c, h, l in zip(closes, highs, lows)]
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
    signals = [strategy.update(c, h, l) for c, h, l in zip(closes, highs, lows)]
    assert signals[2] == 1  # Buy on crossover
    assert signals[-1] == -1  # Trailing take-profit triggers exit
