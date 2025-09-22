import os
from typing import Tuple

import pandas as pd
from mt5.metrics import TICK_ANOMALIES

PRICE_JUMP_THRESHOLD = float(os.getenv("PRICE_JUMP_THRESHOLD", "0.01"))
MAX_SPREAD = float(os.getenv("MAX_SPREAD", "0.05"))


def sanitize_ticks(df: pd.DataFrame, price_jump_threshold: float = PRICE_JUMP_THRESHOLD, max_spread: float = MAX_SPREAD) -> pd.DataFrame:
    """Remove anomalous tick rows based on timestamp order, spread, and price jumps.

    Parameters
    ----------
    df : pd.DataFrame
        Tick dataframe with Timestamp, Bid, and Ask columns.
    price_jump_threshold : float, optional
        Maximum allowed relative jump in mid-price between consecutive ticks.
    max_spread : float, optional
        Maximum allowed spread between ask and bid prices.
    """
    if df.empty:
        return df

    df = df.sort_values("Timestamp")

    ts_diff = df["Timestamp"].diff().dt.total_seconds()
    monotonic_mask = ts_diff > 0
    monotonic_mask.iloc[0] = True
    spread = df["Ask"] - df["Bid"]
    spread_mask = (spread >= 0) & (spread <= max_spread)
    mid = (df["Ask"] + df["Bid"]) / 2
    price_change = mid.pct_change().abs()
    jump_mask = price_change <= price_jump_threshold
    jump_mask.iloc[0] = True

    TICK_ANOMALIES.labels("non_monotonic").inc((~monotonic_mask).sum())
    TICK_ANOMALIES.labels("spread").inc((~spread_mask).sum())
    TICK_ANOMALIES.labels("price_jump").inc((~jump_mask).sum())

    mask = monotonic_mask & spread_mask & jump_mask
    return df[mask]


__all__ = ["sanitize_ticks"]
