"""Volume-based technical indicators.

This module implements two classic volume indicators used to confirm
price trends:

* **On-Balance Volume (OBV)** accumulates volume based on price
  direction. Rising OBV indicates buying pressure while falling OBV
  signals selling pressure.
* **Money Flow Index (MFI)** combines price and volume to measure the
  strength of money flowing in and out of a security. Values above 50
  suggest buying pressure; values below 50 indicate selling pressure.

Both indicators return a new dataframe with ``obv`` and ``mfi`` columns
for downstream strategies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute OBV and MFI indicators.

    Parameters
    ----------
    df:
        DataFrame containing ``High``, ``Low``, ``Volume`` and either
        ``Close`` or ``mid`` price columns.

    Returns
    -------
    pandas.DataFrame
        Original dataframe with additional ``obv`` and ``mfi`` columns.
    """

    df = df.copy()
    price = df.get("Close", df.get("mid"))
    if price is None:
        raise KeyError("DataFrame must contain 'Close' or 'mid' column")

    # On-Balance Volume
    change = price.diff().fillna(0)
    direction = np.sign(change)
    df["obv"] = (direction * df["Volume"]).cumsum()

    # Money Flow Index
    typical = (df["High"] + df["Low"] + price) / 3
    money_flow = typical * df["Volume"]
    pos_flow = money_flow.where(typical > typical.shift(1), 0.0)
    neg_flow = money_flow.where(typical < typical.shift(1), 0.0)
    pos_sum = pos_flow.rolling(14).sum()
    neg_sum = neg_flow.rolling(14).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    df["mfi"] = 100 - 100 / (1 + mfr)

    return df


__all__ = ["compute"]
