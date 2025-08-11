"""Numba-accelerated routines for computational hot spots."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):  # type: ignore
        def wrap(func):
            return func
        return wrap


@njit(cache=True)
def rolling_mean(a: np.ndarray, window: int) -> np.ndarray:
    n = len(a)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    s = 0.0
    for i in range(n):
        val = a[i]
        s += val
        if i >= window:
            s -= a[i - window]
        if i >= window - 1:
            out[i] = s / window
    return out


@njit(cache=True)
def rolling_std(a: np.ndarray, window: int) -> np.ndarray:
    n = len(a)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    sum_x = 0.0
    sum_x2 = 0.0
    for i in range(n):
        val = a[i]
        sum_x += val
        sum_x2 += val * val
        if i >= window:
            old = a[i - window]
            sum_x -= old
            sum_x2 -= old * old
        if i >= window - 1:
            mean = sum_x / window
            var = (sum_x2 - window * mean * mean) / (window - 1)
            out[i] = np.sqrt(var) if var > 0 else 0.0
    return out


@njit(cache=True)
def atr(mid: np.ndarray, window: int) -> np.ndarray:
    n = len(mid)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = 0.0
    for i in range(1, n):
        diff = mid[i] - mid[i - 1]
        tr[i] = diff if diff >= 0 else -diff
    out = rolling_mean(tr, window)
    for i in range(window):
        out[i] = np.nan
    return out


@njit(cache=True)
def rsi(values: np.ndarray, period: int) -> np.ndarray:
    n = len(values)
    delta = np.empty(n, dtype=np.float64)
    delta[0] = 0.0
    for i in range(1, n):
        delta[i] = values[i] - values[i - 1]
    up = np.empty(n, dtype=np.float64)
    down = np.empty(n, dtype=np.float64)
    for i in range(n):
        d = delta[i]
        if d > 0:
            up[i] = d
            down[i] = 0.0
        else:
            up[i] = 0.0
            down[i] = -d
    roll_up = rolling_mean(up, period)
    roll_down = rolling_mean(down, period)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    for i in range(n):
        if i >= period:
            if roll_down[i] == 0.0:
                out[i] = 100.0
            else:
                rs = roll_up[i] / roll_down[i]
                out[i] = 100 - (100 / (1 + rs))
    return out


__all__ = ["rolling_mean", "rolling_std", "atr", "rsi"]
