import os
import sys

# Ensure repo root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from strategies.baseline import BaselineStrategy


def test_trailing_stop_exit_long():
    strategy = BaselineStrategy(
        short_window=2,
        long_window=3,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        trailing_stop_pct=0.05,
        trailing_take_profit_pct=0.05,
    )
    prices = [1, 2, 3, 2.9, 2.5]
    signals = [strategy.update(p) for p in prices]
    assert signals[2] == 1  # Buy on crossover
    assert signals[-1] == -1  # Exit via trailing stop


def test_trailing_take_profit_exit():
    strategy = BaselineStrategy(
        short_window=2,
        long_window=3,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        trailing_stop_pct=0.05,
        trailing_take_profit_pct=0.05,
    )
    prices = [1, 2, 3, 3.5, 3.6, 3.3]
    signals = [strategy.update(p) for p in prices]
    assert signals[2] == 1  # Buy on crossover
    assert signals[-1] == -1  # Trailing take-profit triggers exit
