"""Price-indicator divergence features.

This module computes simple bullish and bearish divergences between
price and momentum oscillators such as RSI or MACD.  A bullish divergence
occurs when price makes a lower low while the indicator makes a higher
low.  Conversely a bearish divergence is flagged when price prints a
higher high but the indicator records a lower high.  The resulting
signals take values ``1`` for bullish, ``-1`` for bearish and ``0`` when
no divergence is detected.
"""

from __future__ import annotations

import pandas as pd


def compute_divergence(df: pd.DataFrame, indicator: str) -> pd.Series:
    """Return divergence signal between ``indicator`` and price.

    Parameters
    ----------
    df:
        DataFrame containing a price column (``Close`` or ``mid``) and the
        indicator column.
    indicator:
        Name of the column containing the oscillator values (e.g. ``rsi``
        or ``macd``).

    Returns
    -------
    pandas.Series
        Series of ``1`` for bullish divergence, ``-1`` for bearish and ``0``
        otherwise.  If required columns are missing a zero series is
        returned.
    """

    price = df.get("Close", df.get("mid"))
    if price is None or indicator not in df:
        return pd.Series(0, index=df.index, dtype=int)

    ind = df[indicator]
    div = pd.Series(0, index=df.index, dtype=int)
    for i in range(1, len(df)):
        p_prev, p_curr = price.iloc[i - 1], price.iloc[i]
        i_prev, i_curr = ind.iloc[i - 1], ind.iloc[i]
        if p_curr > p_prev and i_curr < i_prev:
            div.iloc[i] = -1
        elif p_curr < p_prev and i_curr > i_prev:
            div.iloc[i] = 1
    return div


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Append RSI and MACD divergence columns when available."""
    df = df.copy()
    if "rsi" in df.columns:
        df["div_rsi"] = compute_divergence(df, "rsi")
    if "macd" in df.columns:
        df["div_macd"] = compute_divergence(df, "macd")
    return df


__all__ = ["compute", "compute_divergence"]
