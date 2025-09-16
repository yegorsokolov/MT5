"""Volume-weighted average price (VWAP) features.

This module computes VWAP anchored to the current trading session and
calendar day.  It also returns a simple crossover signal indicating
whether the session VWAP is above or below the day VWAP which can be
used to align strategies with intraday trend direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.session_features import classify_session
from indicators.common import vwap as calc_vwap


def compute(df: pd.DataFrame, anchors=("session", "day")) -> pd.DataFrame:
    """Compute session and daily VWAP along with a crossover signal.

    Parameters
    ----------
    df:
        DataFrame containing ``Timestamp``, ``Volume`` and either ``Close``
        or ``mid`` price columns.
    anchors:
        Iterable specifying which VWAP anchors to compute. Supported
        values are ``"session"`` and ``"day"``.

    Returns
    -------
    pandas.DataFrame
        Original dataframe with added ``vwap_session``, ``vwap_day`` and
        ``vwap_cross`` columns. ``vwap_cross`` is ``1`` when session VWAP
        is above day VWAP, ``-1`` when below and ``0`` otherwise.
    """

    df = df.copy()
    price = df.get("Close", df.get("mid"))
    if price is None:
        raise KeyError("DataFrame must contain 'Close' or 'mid' column")
    volume = df["Volume"]
    times = pd.to_datetime(df["Timestamp"], utc=True)

    if "session" in anchors:
        sessions = times.map(classify_session).fillna("other")
        group = list(zip(times.dt.date, sessions))
        df["vwap_session"] = calc_vwap(price, volume, group)

    if "day" in anchors:
        day_group = times.dt.date
        df["vwap_day"] = calc_vwap(price, volume, day_group)

    if {"session", "day"}.issubset(anchors):
        df["vwap_cross"] = np.sign(df["vwap_session"] - df["vwap_day"]).fillna(0).astype(int)
    else:
        df["vwap_cross"] = 0

    return df


__all__ = ["compute"]
