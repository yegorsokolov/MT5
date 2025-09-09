from __future__ import annotations

"""Moving-average crossover baseline strategy.

This strategy maintains short- and long-term simple moving averages over a
stream of prices. A buy order is generated when the short-term average crosses
above the long-term average. A sell order is generated when the short-term
average crosses below the long-term average. Orders are expressed as
``{"quantity": 1}`` for buy and ``{"quantity": -1}`` for sell. When no
crossover occurs, the quantity is ``0``.
"""

from collections import deque
from typing import Deque, Dict, Any


class BaselineMovingAverageStrategy:
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

    def generate_order(self, tick: Dict[str, Any]) -> Dict[str, int]:
        """Generate an order based on the latest price tick."""

        price = float(tick.get("price", 0))
        self._short.append(price)
        self._long.append(price)
        if len(self._long) < self.long_window:
            return {"quantity": 0}

        short_ma = sum(self._short) / self.short_window
        long_ma = sum(self._long) / self.long_window

        quantity = 0
        if short_ma > long_ma and self._prev_short <= self._prev_long:
            quantity = 1
        elif short_ma < long_ma and self._prev_short >= self._prev_long:
            quantity = -1

        self._prev_short, self._prev_long = short_ma, long_ma
        return {"quantity": quantity}

    def update(self, order: Dict[str, Any], outcome: float) -> None:  # pragma: no cover - simple baseline
        """Update internal state from trade outcome (no-op for baseline)."""
        return
