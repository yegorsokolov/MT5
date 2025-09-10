"""Order flow based features.

This module derives microstructural signals from top-of-book or tick data:

* **Volume imbalance** between bid and ask sides.
* **Cumulative volume delta (CVD)** capturing buying vs. selling pressure.
* Short term rolling statistics of both metrics for quick trend assessment.

The function is robust to varying column conventions. It searches for common
bid/ask volume column names such as ``bid_sz_0``/``ask_sz_0`` or
``BidVolume1``/``AskVolume1``. When no suitable columns are found the input
is returned unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # pragma: no cover - decorator optional in standalone tests
    from . import validate_module
except Exception:  # pragma: no cover - fallback when imported directly
    def validate_module(func):
        return func


_BID_CANDIDATES = [
    "bid_volume",
    "bid_vol",
    "BidVolume1",
    "BidVolume0",
    "bid_sz_0",
    "bid_size",
]
_ASK_CANDIDATES = [
    "ask_volume",
    "ask_vol",
    "AskVolume1",
    "AskVolume0",
    "ask_sz_0",
    "ask_size",
]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


@validate_module
def compute(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Compute order flow features.

    Parameters
    ----------
    df:
        DataFrame containing bid/ask volume or size columns.
    window:
        Rolling window size for short term statistics. Default is ``10``.

    Returns
    -------
    pandas.DataFrame
        Original dataframe with ``cvd`` and ``imbalance`` columns along with
        rolling mean and standard deviation of both metrics.
    """

    bid_col = _find_col(df, _BID_CANDIDATES)
    ask_col = _find_col(df, _ASK_CANDIDATES)
    if bid_col is None or ask_col is None:
        return df

    df = df.copy()
    bid = df[bid_col]
    ask = df[ask_col]
    delta = bid - ask
    total = bid + ask

    df["imbalance"] = (delta / total.replace(0, np.nan)).fillna(0.0)
    df["cvd"] = delta.cumsum()

    for col in ["imbalance", "cvd"]:
        roll = df[col].rolling(window)
        df[f"{col}_roll_mean"] = roll.mean()
        df[f"{col}_roll_std"] = roll.std()

    return df


__all__ = ["compute"]
