from __future__ import annotations

"""Simple Kalman filter utilities."""

import numpy as np
import pandas as pd


def kalman_smooth(series: pd.Series, process_var: float = 1e-2, measurement_var: float | None = None) -> pd.DataFrame:
    """Apply a 1D Kalman filter to ``series``.

    Parameters
    ----------
    series : pd.Series
        Observed price series possibly containing noise.
    process_var : float, optional
        Variance of the process noise.  Larger values allow the filter to
        adapt more quickly to changes in the underlying signal.
    measurement_var : float, optional
        Variance of the measurement noise.  If ``None`` the variance of the
        first difference of ``series`` is used.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``price`` containing the filtered price and
        ``volatility`` containing the estimated standard deviation of the
        state.
    """

    series = pd.Series(series).astype(float)
    n = len(series)
    if n == 0:
        return pd.DataFrame(columns=["price", "volatility"])

    if measurement_var is None or measurement_var == 0:
        measurement_var = series.diff().var()
        if measurement_var is None or np.isnan(measurement_var) or measurement_var == 0:
            measurement_var = 1.0

    xhat = np.zeros(n)
    P = np.zeros(n)

    xhat[0] = series.iloc[0]
    P[0] = 1.0

    for k in range(1, n):
        # Predict
        xhatminus = xhat[k - 1]
        Pminus = P[k - 1] + process_var

        # Update
        K = Pminus / (Pminus + measurement_var)
        xhat[k] = xhatminus + K * (series.iloc[k] - xhatminus)
        P[k] = (1 - K) * Pminus

    vol = np.sqrt(P)
    return pd.DataFrame({"price": xhat, "volatility": vol}, index=series.index)


__all__ = ["kalman_smooth"]
