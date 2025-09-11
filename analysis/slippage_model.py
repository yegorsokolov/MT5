"""Utilities for estimating order slippage from order book data."""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Sequence, Tuple

import numpy as np


PriceLevel = Tuple[float, float]
Snapshot = Sequence[PriceLevel]


def snapshot_slippage(
    order_size: float,
    side: str,
    levels: Snapshot,
    *,
    best_price: float | None = None,
) -> float:
    """Compute slippage for a single order book snapshot.

    Parameters
    ----------
    order_size: float
        Quantity to execute.
    side: str
        ``"buy"`` or ``"sell"`` to determine sign.
    levels: Sequence[Tuple[float, float]]
        Price/volume pairs ordered from best to worst price for the given side.
    best_price: float, optional
        Explicit best price.  When omitted the first level price is used.
    Returns
    -------
    float
        Expected slippage in basis points relative to ``best_price``.
    """
    if order_size <= 0:
        return 0.0
    if not levels:
        return 0.0
    best = best_price if best_price is not None else levels[0][0]
    remaining = order_size
    cost = 0.0
    for price, vol in levels:
        take = min(remaining, vol)
        cost += price * take
        remaining -= take
        if remaining <= 0:
            break
    if remaining > 0:
        # assume last price for any remaining quantity
        cost += levels[-1][0] * remaining
    vwap = cost / order_size
    sign = 1 if side.lower() == "buy" else -1
    return (vwap - best) / best * sign * 10000.0


class SlippageModel:
    """Estimate expected slippage from historical order book snapshots."""

    def __init__(self, history: Iterable[Snapshot]):
        self.history: Deque[Snapshot] = deque([list(s) for s in history])

    def __call__(self, order_size: float, side: str) -> float:
        if not self.history:
            return 0.0
        slips = [snapshot_slippage(order_size, side, snap) for snap in self.history]
        return float(np.mean(slips))
