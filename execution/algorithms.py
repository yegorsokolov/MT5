"""Execution scheduling algorithms.

This module provides simple time and volume weighted execution strategies
used by :class:`ExecutionEngine` to break up large parent orders into a
sequence of child orders.
"""
from __future__ import annotations

import asyncio
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


# ---------------------------------------------------------------------------
async def twap_schedule_async(total_qty: float, intervals: int) -> List[float]:
    """Async wrapper around :func:`twap_schedule`.

    The helper mirrors :func:`twap_schedule` but provides an ``await`` point so
    callers can schedule slice generation without blocking the event loop.
    ``asyncio.sleep(0)`` yields control and therefore keeps behaviour
    deterministic for tests.
    """

    await asyncio.sleep(0)
    return twap_schedule(total_qty, intervals)


async def vwap_schedule_async(total_qty: float, volumes: Iterable[float]) -> List[float]:
    """Async wrapper around :func:`vwap_schedule`.

    Similar to :func:`twap_schedule_async`, this simply delegates to the
    synchronous implementation after yielding to the event loop.  Having an
    asynchronous entry point keeps the scheduling API uniform for synchronous
    and asynchronous execution engines.
    """

    await asyncio.sleep(0)
    return vwap_schedule(total_qty, volumes)

