"""Kalman-filter-based moving average (KMA).

This module implements a simple 1D Kalman filter to smooth the closing
price and generate a Kalman moving average (`kma`).  The filter
maintains a state estimate `x` and its variance `P` and updates them as
new observations arrive.  Two hyperparameters control the behaviour of
the filter:

Parameters
----------
process_noise : float, default 1e-5
    Variance of the process noise (``Q``).  Higher values allow the
    estimate to adapt faster at the expense of increased noise.
measurement_noise : float, default 1e-2
    Variance of the observation noise (``R``).  Higher values result in
    stronger smoothing and therefore more lag.

The :func:`compute` function adds two columns to the input dataframe:
``kma`` containing the filtered price and ``kma_cross`` which signals
1 when price crosses above ``kma`` and -1 for the opposite.
"""

from __future__ import annotations

import pandas as pd


def _kalman_filter(
    series: pd.Series, process_noise: float, measurement_noise: float
) -> pd.Series:
    """Return the Kalman moving average for ``series``."""
    if series.empty:
        return series.copy()

    x = float(series.iloc[0])
    p = 1.0
    estimates = [x]

    for z in series.iloc[1:]:
        # Prediction step: state is assumed to persist
        p = p + process_noise
        # Update step
        k = p / (p + measurement_noise)
        x = x + k * (float(z) - x)
        p = (1 - k) * p
        estimates.append(x)

    return pd.Series(estimates, index=series.index)


def compute(
    df: pd.DataFrame, process_noise: float = 1e-5, measurement_noise: float = 1e-2
) -> pd.DataFrame:
    """Compute Kalman moving average and price cross signals."""
    df = df.copy()
    price = df["Close"]
    kma = _kalman_filter(price, process_noise, measurement_noise)
    df["kma"] = kma
    cross = ((price > kma) & (price.shift(1) <= kma.shift(1))).astype(int) - (
        (price < kma) & (price.shift(1) >= kma.shift(1))
    ).astype(int)
    df["kma_cross"] = cross.fillna(0).astype(int)
    return df


__all__ = ["compute"]
