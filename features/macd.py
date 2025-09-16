"""Moving Average Convergence Divergence (MACD) features.

This module computes the classic MACD oscillator consisting of the
MACD line (fast EMA minus slow EMA), the signal line (EMA of the MACD
line) and a simple directional indicator ``macd_cross``.

``macd_cross`` is ``1`` when the MACD line is above the signal line,
``-1`` when below and ``0`` otherwise.  It can be used by strategies to
require bullish or bearish confirmation from MACD before acting on other
signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.common import macd as calc_macd


def compute(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Compute MACD, signal line and crossover direction.

    Parameters
    ----------
    df:
        DataFrame containing ``Close`` prices or ``mid`` prices.
    fast, slow, signal:
        Window lengths for the fast EMA, slow EMA and signal line
        respectively.

    Returns
    -------
    pandas.DataFrame
        Original dataframe with added ``macd``, ``macd_signal`` and
        ``macd_cross`` columns.
    """

    df = df.copy()
    price = df.get("Close", df.get("mid"))
    if price is None:
        raise KeyError("DataFrame must contain 'Close' or 'mid' column")

    macd_line, signal_line, hist = calc_macd(price, fast=fast, slow=slow, signal=signal)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_cross"] = np.sign(hist).fillna(0).astype(int)
    return df


__all__ = ["compute"]
