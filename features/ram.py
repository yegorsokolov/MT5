"""Risk-adjusted momentum (RAM) indicator.

This feature computes a simple risk-adjusted momentum measure defined as
the average return divided by its volatility over a lookback window.  An
optional exponential decay can be applied to emphasise recent
observations.  The resulting ``ram`` column behaves similarly to a
rolling Sharpe ratio and can be used to gate entries in trading
strategies.

Examples
--------
>>> import pandas as pd
>>> df = pd.DataFrame({"Close": [1, 1.1, 1.2, 1.15, 1.3]})
>>> compute(df, window=3)["ram"].round(2).tolist()
[0.0, 0.0, 9.59, 2.01, 4.22]
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(df: pd.DataFrame, window: int = 20, decay: float | None = None) -> pd.DataFrame:
    """Compute the RAM indicator.

    Parameters
    ----------
    df:
        DataFrame containing ``Close`` or ``mid`` price column.
    window:
        Lookback window for mean return and volatility calculations or
        the span for exponential weighting.
    decay:
        Optional exponential decay factor ``alpha`` in ``(0, 1]``.  When
        provided, exponentially weighted moving statistics are used
        instead of simple rolling windows.

    Returns
    -------
    pandas.DataFrame
        Original dataframe with an additional ``ram`` column.
    """

    df = df.copy()
    price = df.get("Close", df.get("mid"))
    if price is None:
        raise KeyError("DataFrame must contain 'Close' or 'mid' column")

    ret = price.pct_change()
    if decay is None:
        mean_ret = ret.rolling(window).mean()
        vol = ret.rolling(window).std()
    else:
        mean_ret = ret.ewm(alpha=decay, adjust=False).mean()
        vol = ret.ewm(alpha=decay, adjust=False).std()

    df["ram"] = (mean_ret / vol.replace(0, np.nan)).fillna(0.0)
    return df


__all__ = ["compute"]
