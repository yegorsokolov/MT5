"""Helpers for checking whether a market is currently open."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd


def is_market_open(exchange: str = "24/5", now: Optional[datetime] = None) -> bool:
    """Return ``True`` if the given exchange calendar is open.

    Parameters
    ----------
    exchange:
        Identifier understood by :mod:`exchange_calendars` such as ``"XNYS"``
        or ``"24/5"`` for around-the-clock markets.
    now:
        Time to test. Defaults to the current UTC time.
    """
    try:  # exchange_calendars is an optional dependency
        from exchange_calendars import get_calendar
    except Exception:  # pragma: no cover - fallback if package missing
        ts = now or datetime.utcnow()
        return ts.weekday() < 5

    ts = pd.Timestamp.utcnow() if now is None else pd.Timestamp(now, tz="UTC")
    cal = get_calendar(exchange)
    return bool(cal.is_open_at_time(ts))


__all__ = ["is_market_open"]
