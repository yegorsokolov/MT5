"""Utilities for encoding major market sessions.

This module provides helpers to classify timestamps into major
trading sessions (Tokyo, London, New York) and generate one-hot or
cyclical encodings. Time zone boundaries and daylight saving time
shifts are handled automatically using :mod:`zoneinfo`.
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

# Definition of major market sessions. Times are given in the
# respective local time zones so DST transitions are handled by
# ``zoneinfo`` when converting from UTC timestamps.
SESSION_DEFS: Dict[str, Dict[str, object]] = {
    # Order reflects priority when sessions overlap
    "london": {"tz": ZoneInfo("Europe/London"), "start": time(8, 0), "end": time(16, 0)},
    "new_york": {"tz": ZoneInfo("America/New_York"), "start": time(8, 0), "end": time(17, 0)},
    "tokyo": {"tz": ZoneInfo("Asia/Tokyo"), "start": time(9, 0), "end": time(17, 0)},
}


def classify_session(ts: pd.Timestamp) -> Optional[str]:
    """Return the active session name for a timestamp.

    Parameters
    ----------
    ts:
        Timestamp assumed to be in UTC.

    Returns
    -------
    Optional[str]
        Name of the active session (tokyo, london, new_york) or ``None`` if
        the timestamp falls outside all sessions.
    """

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    for name, spec in SESSION_DEFS.items():
        tz: ZoneInfo = spec["tz"]  # type: ignore[assignment]
        local = ts.astimezone(tz)
        start: time = spec["start"]  # type: ignore[assignment]
        end: time = spec["end"]  # type: ignore[assignment]
        lt = local.timetz().replace(tzinfo=None)
        if start <= lt < end:
            return name
    return None


def session_onehot(times: Iterable[pd.Timestamp]) -> pd.DataFrame:
    """Generate one-hot encodings for each session.

    Parameters
    ----------
    times:
        Iterable of timestamps in UTC.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``session_tokyo``, ``session_london`` and
        ``session_new_york`` containing binary indicators for the active
        session.
    """

    times = pd.to_datetime(list(times), utc=True)
    data = {f"session_{name}": [] for name in SESSION_DEFS}
    for ts in times:
        active = classify_session(ts)
        for name in SESSION_DEFS:
            data[f"session_{name}"].append(int(active == name))
    return pd.DataFrame(data, index=getattr(times, "index", None))


def session_cyclical(times: Iterable[pd.Timestamp]) -> pd.DataFrame:
    """Generate cyclical encodings (sine and cosine) for sessions."""

    times = pd.to_datetime(list(times), utc=True)
    out: Dict[str, List[float]] = {}
    for name, spec in SESSION_DEFS.items():
        tz: ZoneInfo = spec["tz"]  # type: ignore[assignment]
        start: time = spec["start"]  # type: ignore[assignment]
        end: time = spec["end"]  # type: ignore[assignment]
        start_dt = datetime.combine(datetime(2000, 1, 1), start)
        end_dt = datetime.combine(datetime(2000, 1, 1), end)
        length = (end_dt - start_dt).total_seconds()
        sin_vals = []
        cos_vals = []
        for ts in times:
            local = ts.astimezone(tz)
            lt = local.timetz().replace(tzinfo=None)
            if start <= lt < end:
                delta = datetime.combine(local.date(), lt) - datetime.combine(
                    local.date(), start
                )
                angle = 2 * np.pi * delta.total_seconds() / length
                sin_vals.append(np.sin(angle))
                cos_vals.append(np.cos(angle))
            else:
                sin_vals.append(0.0)
                cos_vals.append(0.0)
        out[f"{name}_sin"] = sin_vals
        out[f"{name}_cos"] = cos_vals
    return pd.DataFrame(out, index=getattr(times, "index", None))


def add_session_features(df: pd.DataFrame, cyclical: bool = False) -> pd.DataFrame:
    """Add session encodings to a DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        Must contain a ``Timestamp`` column in UTC.
    cyclical: bool, default ``False``
        If ``True`` also include sine/cosine cyclical encodings.
    """

    if "Timestamp" not in df.columns:
        raise KeyError("Timestamp column required for session features")
    times = pd.to_datetime(df["Timestamp"], utc=True)
    df = df.copy()
    df = df.join(session_onehot(times))
    if cyclical:
        df = df.join(session_cyclical(times))
    return df


__all__ = [
    "SESSION_DEFS",
    "classify_session",
    "session_onehot",
    "session_cyclical",
    "add_session_features",
]
