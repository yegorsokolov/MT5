"""Microprice-based order book features.

This module derives a microprice signal from top-of-book quotes. The
microprice skews the mid price toward the side with greater liquidity and
thus provides a simple measure of buying or selling pressure.  In addition
to the raw microprice this module also computes ``microprice_delta`` – the
difference between the microprice and the traditional mid price.  A
positive ``microprice_delta`` indicates buy-side pressure while a negative
value indicates sell-side pressure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Candidate column names for flexibility across datasets
_BID_PX_CANDIDATES = ["bid_px_0", "BidPrice0", "BidPrice1", "Bid"]
_ASK_PX_CANDIDATES = ["ask_px_0", "AskPrice0", "AskPrice1", "Ask"]
_BID_SZ_CANDIDATES = ["bid_sz_0", "BidVolume0", "BidVolume1", "bid_size", "bid_volume"]
_ASK_SZ_CANDIDATES = ["ask_sz_0", "AskVolume0", "AskVolume1", "ask_size", "ask_volume"]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column present in ``df`` from ``candidates``."""

    for c in candidates:
        if c in df.columns:
            return c
    return None


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute microprice and buy/sell pressure from top-of-book quotes.

    Parameters
    ----------
    df:
        DataFrame containing best bid/ask prices and sizes.

    Returns
    -------
    pandas.DataFrame
        Original dataframe augmented with ``microprice`` and
        ``microprice_delta`` columns. If required columns are missing the
        input dataframe is returned unchanged.
    """

    bid_px_col = _find_col(df, _BID_PX_CANDIDATES)
    ask_px_col = _find_col(df, _ASK_PX_CANDIDATES)
    bid_sz_col = _find_col(df, _BID_SZ_CANDIDATES)
    ask_sz_col = _find_col(df, _ASK_SZ_CANDIDATES)
    if None in (bid_px_col, ask_px_col, bid_sz_col, ask_sz_col):
        return df

    df = df.copy()
    bid_px = df[bid_px_col]
    ask_px = df[ask_px_col]
    bid_sz = df[bid_sz_col]
    ask_sz = df[ask_sz_col]

    mid = (bid_px + ask_px) / 2
    total = bid_sz + ask_sz
    microprice = (ask_px * bid_sz + bid_px * ask_sz) / total.replace(0, np.nan)
    microprice = microprice.fillna(mid)

    df["microprice"] = microprice
    df["microprice_delta"] = microprice - mid
    return df


__all__ = ["compute"]

