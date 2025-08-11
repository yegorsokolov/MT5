from __future__ import annotations

import numpy as np
import pandas as pd


def garch_volatility(series: pd.Series) -> pd.Series:
    """Estimate conditional volatility using an EGARCH(1,1) model.

    Parameters
    ----------
    series : pd.Series
        Return series. Should be numeric and typically represent percent
        changes of a price series.

    Returns
    -------
    pd.Series
        Estimated one-step-ahead conditional volatility. If the ``arch``
        library is unavailable or the series is too short, a 30-period
        rolling standard deviation is returned instead.
    """

    # Ensure we work with a Series and keep original index
    s = pd.Series(series).astype(float)
    idx = s.index
    s_clean = s.dropna()

    if len(s_clean) < 20:
        return pd.Series(np.nan, index=idx)

    try:
        from arch import arch_model  # type: ignore

        am = arch_model(s_clean * 100, vol="EGARCH", p=1, o=1, q=1, dist="normal")
        res = am.fit(disp="off")
        vol = res.conditional_volatility / 100.0
        vol = vol.reindex(idx)
        return vol
    except Exception:
        # Fallback to simple rolling volatility when ``arch`` is not available
        return s.rolling(30).std()
