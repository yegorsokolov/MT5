"""GPU accelerated feature computations using CuPy."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import pywt

try:  # pragma: no cover - cupy optional
    import cupy as cp
    from cupyx.scipy.signal import convolve
except Exception:  # pragma: no cover - cupy optional
    cp = None  # type: ignore
    convolve = None  # type: ignore

from .garch_vol import garch_volatility as _cpu_garch_volatility
from .frequency_features import (
    rolling_fft_features as _cpu_rolling_fft_features,
    rolling_wavelet_features as _cpu_rolling_wavelet_features,
)
from .fractal_features import (
    rolling_fractal_features as _cpu_rolling_fractal_features,
)


def _to_gpu(series: pd.Series | np.ndarray) -> tuple["cp.ndarray", pd.Index | None]:
    """Convert ``series`` to a CuPy array returning index if present."""
    if cp is None:
        raise RuntimeError("CuPy is required for GPU features")
    if isinstance(series, pd.Series):
        index = series.index
        arr = series.to_numpy()
    else:
        arr = np.asarray(series)
        index = None
    return cp.asarray(arr, dtype=cp.float64), index


def garch_volatility_gpu(series: pd.Series) -> pd.Series:
    """Estimate volatility using a simple GARCH(1,1) process on the GPU."""
    if cp is None:
        return _cpu_garch_volatility(series)
    arr, index = _to_gpu(series)
    omega = cp.float64(0.1)
    alpha = cp.float64(0.1)
    beta = cp.float64(0.8)
    var = cp.zeros_like(arr)
    for i in range(1, arr.size):
        var[i] = omega + alpha * arr[i - 1] ** 2 + beta * var[i - 1]
    vol = cp.sqrt(var)
    return pd.Series(cp.asnumpy(vol), index=index)


def rolling_fft_features_gpu(
    series: pd.Series | np.ndarray,
    window: int = 128,
    freqs: Iterable[float] = (0.01,),
) -> pd.DataFrame:
    """CuPy powered FFT magnitude extraction."""
    if cp is None:
        return _cpu_rolling_fft_features(series, window=window, freqs=freqs)
    arr, index = _to_gpu(series)
    n = arr.size
    freqs = list(freqs)
    res = {f"fft_mag_{f}": cp.full(n, cp.nan) for f in freqs}
    for end in range(window, n + 1):
        segment = arr[end - window : end]
        fft_vals = cp.abs(cp.fft.rfft(segment))
        fft_freqs = cp.fft.rfftfreq(window, d=1)
        for f in freqs:
            idx = cp.argmin(cp.abs(fft_freqs - f))
            res[f"fft_mag_{f}"][end - 1] = fft_vals[idx]
    res_np = {k: cp.asnumpy(v) for k, v in res.items()}
    return pd.DataFrame(res_np, index=index)


def rolling_wavelet_features_gpu(
    series: pd.Series | np.ndarray,
    window: int = 128,
    wavelet: str = "db4",
    level: int = 2,
) -> pd.DataFrame:
    """Compute wavelet detail coefficients on GPU using CuPy convolutions."""
    if cp is None or convolve is None:
        return _cpu_rolling_wavelet_features(series, window=window, wavelet=wavelet, level=level)
    if isinstance(series, pd.Series):
        index = series.index
        arr = cp.asarray(series.to_numpy(), dtype=cp.float64)
    else:
        index = None
        arr = cp.asarray(series, dtype=cp.float64)
    n = arr.size
    name = f"wavelet_{wavelet}_lvl{level}"
    res = cp.full(n, cp.nan)
    w = pywt.Wavelet(wavelet)
    dec_lo = cp.asarray(w.dec_lo)
    dec_hi = cp.asarray(w.dec_hi)
    for end in range(window, n + 1):
        segment = arr[end - window : end]
        approx = segment
        for _ in range(level - 1):
            approx = convolve(approx, dec_lo, mode="valid")[::2]
        detail = convolve(approx, dec_hi, mode="valid")[::2]
        res[end - 1] = cp.mean(cp.abs(detail))
    return pd.DataFrame({name: cp.asnumpy(res)}, index=index)


def rolling_fractal_features_gpu(
    series: pd.Series | np.ndarray,
    window: int = 128,
) -> pd.DataFrame:
    """Compute rolling fractal features using CuPy."""
    if cp is None:
        return _cpu_rolling_fractal_features(series, window=window)
    if isinstance(series, pd.Series):
        index = series.index
        arr = cp.asarray(series.to_numpy(), dtype=cp.float64)
    else:
        index = None
        arr = cp.asarray(series, dtype=cp.float64)
    n = arr.size
    hurst = cp.full(n, cp.nan)
    fd = cp.full(n, cp.nan)
    log_n = cp.log10(cp.float64(window))
    for end in range(window, n + 1):
        segment = arr[end - window : end]
        L = cp.sum(cp.abs(cp.diff(segment)))
        d = cp.max(cp.abs(segment - segment[0]))
        if L == 0 or d == 0:
            fdim = cp.float64(1.0)
        else:
            fdim = log_n / (log_n + cp.log10(d / L))
        fd[end - 1] = fdim
        hurst[end - 1] = 1.0 / fdim
    data = {"hurst": cp.asnumpy(hurst), "fractal_dim": cp.asnumpy(fd)}
    return pd.DataFrame(data, index=index)


# Mark these functions as requiring GPU resources
for _fn in [
    garch_volatility_gpu,
    rolling_fft_features_gpu,
    rolling_wavelet_features_gpu,
    rolling_fractal_features_gpu,
]:
    _fn.min_capability = "gpu"  # type: ignore[attr-defined]
