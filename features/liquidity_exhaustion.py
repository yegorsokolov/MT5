"""Liquidity exhaustion feature.

This module computes the distribution of bid and ask depth close to the mid
price.  The relative depth provides a simple gauge of which side of the book
is likely to exhaust first.  Two signals are produced:

``liq_ratio``
    Ratio of bid depth to ask depth within ``ticks`` of the mid price.
``liq_exhaustion``
    Discrete signal derived from ``liq_ratio``. ``1`` indicates the ask side is
    thin relative to bids (potential upward pressure) while ``-1`` indicates the
    bid side is thin (potential downward pressure).

The input ``DataFrame`` must contain level based order book data with columns
following the pattern ``bid_px_<i>``, ``bid_sz_<i>``, ``ask_px_<i>`` and
``ask_sz_<i>`` where ``i`` ranges from ``0`` to ``depth-1``.
"""

from __future__ import annotations

import pandas as pd


def _infer_depth(df: pd.DataFrame) -> int:
    depth = 0
    while (
        f"bid_px_{depth}" in df.columns
        and f"bid_sz_{depth}" in df.columns
        and f"ask_px_{depth}" in df.columns
        and f"ask_sz_{depth}" in df.columns
    ):
        depth += 1
    return depth


def compute(
    df: pd.DataFrame,
    ticks: int = 3,
    upper_ratio: float = 2.0,
    lower_ratio: float = 0.5,
) -> pd.DataFrame:
    """Compute liquidity exhaustion metrics.

    Parameters
    ----------
    df:
        DataFrame containing bid/ask price and size columns.
    ticks:
        Number of price ticks away from the mid price to accumulate depth.
    upper_ratio:
        Threshold above which ``liq_exhaustion`` is ``1``.
    lower_ratio:
        Threshold below which ``liq_exhaustion`` is ``-1``.

    Returns
    -------
    pandas.DataFrame
        Original dataframe with ``liq_ratio`` and ``liq_exhaustion`` columns.
        If required columns are missing the input dataframe is returned
        unchanged.
    """

    depth = _infer_depth(df)
    if depth == 0:
        return df

    df = df.copy()
    bid0 = df["bid_px_0"]
    ask0 = df["ask_px_0"]
    mid = (bid0 + ask0) / 2
    tick_size = (ask0 - bid0).replace(0, pd.NA)
    max_dist = ticks * tick_size

    bid_depth = pd.Series(0.0, index=df.index)
    ask_depth = pd.Series(0.0, index=df.index)

    for i in range(depth):
        bp = df[f"bid_px_{i}"]
        bs = df[f"bid_sz_{i}"]
        ap = df[f"ask_px_{i}"]
        az = df[f"ask_sz_{i}"]
        bid_depth += bs.where(mid - bp <= max_dist, 0.0)
        ask_depth += az.where(ap - mid <= max_dist, 0.0)

    liq_ratio = bid_depth / ask_depth.replace(0, pd.NA)
    df["liq_ratio"] = liq_ratio.fillna(0.0)
    df["liq_exhaustion"] = 0
    df.loc[df["liq_ratio"] > upper_ratio, "liq_exhaustion"] = 1
    df.loc[df["liq_ratio"] < lower_ratio, "liq_exhaustion"] = -1
    return df


__all__ = ["compute"]
