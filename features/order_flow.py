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

try:  # pragma: no cover - polars optional
    import polars as pl
except Exception:  # pragma: no cover
    pl = None  # type: ignore

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


def _find_col(df, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


@validate_module
def compute(df, window: int = 10):
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

    if pl is not None and isinstance(df, pl.DataFrame):
        delta = pl.col(bid_col) - pl.col(ask_col)
        total = pl.col(bid_col) + pl.col(ask_col)
        df = df.with_columns(
            [
                (delta / total.replace(0, None)).fill_null(0.0).alias("imbalance"),
                delta.cum_sum().alias("cvd"),
            ]
        )
        df = df.with_columns(
            [
                pl.col("imbalance").rolling_mean(window).alias("imbalance_roll_mean"),
                pl.col("imbalance").rolling_std(window).alias("imbalance_roll_std"),
                pl.col("cvd").rolling_mean(window).alias("cvd_roll_mean"),
                pl.col("cvd").rolling_std(window).alias("cvd_roll_std"),
            ]
        )
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


compute.supports_polars = True  # type: ignore[attr-defined]


__all__ = ["compute"]
