"""Minimal moving-average crossover baseline strategy.

This strategy is intentionally lightweight so it can run even when
system resources are scarce. It maintains short and long moving
averages over a stream of prices and produces trading signals when the
short average crosses the long average.
"""
from collections import deque
from typing import Deque, List


class BaselineStrategy:
    """Simple moving-average crossover strategy.

    Parameters
    ----------
    short_window: int
        Number of recent prices for the fast moving average.
    long_window: int
        Number of recent prices for the slow moving average.
    """

    def __init__(self, short_window: int = 5, long_window: int = 20) -> None:
        if short_window >= long_window:
            raise ValueError("short_window must be < long_window")
        self.short_window = short_window
        self.long_window = long_window
        self._short: Deque[float] = deque(maxlen=short_window)
        self._long: Deque[float] = deque(maxlen=long_window)
        self._prev_short = 0.0
        self._prev_long = 0.0

    def update(self, price: float) -> int:
        """Process a new price and return a trading signal.

        Returns
        -------
        int
            1 for buy, -1 for sell, 0 for hold.
        """

        self._short.append(price)
        self._long.append(price)
        if len(self._long) < self.long_window:
            # Not enough data yet
            return 0

        short_ma = sum(self._short) / self.short_window
        long_ma = sum(self._long) / self.long_window

        signal = 0
        if short_ma > long_ma and self._prev_short <= self._prev_long:
            signal = 1
        elif short_ma < long_ma and self._prev_short >= self._prev_long:
            signal = -1

        self._prev_short, self._prev_long = short_ma, long_ma
        return signal
