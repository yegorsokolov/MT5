"""Execution scheduling algorithms.

This module provides simple time and volume weighted execution strategies
used by :class:`ExecutionEngine` to break up large parent orders into a
sequence of child orders.
"""
from __future__ import annotations

from typing import Iterable, List


def twap_schedule(total_qty: float, intervals: int) -> List[float]:
    """Return a time weighted allocation for ``total_qty``.

    Parameters
    ----------
    total_qty: float
        Total quantity to execute.
    intervals: int
        Number of time slices to split the order into. If ``intervals`` is
        non-positive the entire quantity is returned in a single slice.
    """
    if intervals <= 0:
        return [total_qty]
    slice_size = total_qty / intervals
    return [slice_size for _ in range(intervals)]


def vwap_schedule(total_qty: float, volumes: Iterable[float]) -> List[float]:
    """Return a volume weighted allocation for ``total_qty``.

    The ``volumes`` iterable represents recent traded volume for consecutive
    periods. The parent order will be apportioned proportional to these
    volumes. When no volume data is supplied the entire quantity is executed
    immediately.
    """
    vols = [v for v in volumes if v > 0]
    if not vols:
        return [total_qty]
    total_vol = sum(vols)
    if total_vol == 0:
        return [total_qty]
    return [total_qty * v / total_vol for v in vols]
