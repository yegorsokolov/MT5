"""Frequency domain feature extraction utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import pywt


def rolling_fft_features(
    series: pd.Series | np.ndarray,
    window: int = 128,
    freqs: Iterable[float] = (0.01,),
) -> pd.DataFrame:
    """Compute FFT magnitudes over rolling windows.

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Input signal.
    window : int, default 128
        Rolling window size.
    freqs : Iterable[float]
        Target frequencies (cycles per sample) to extract magnitudes for.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``fft_mag_<freq>`` containing magnitudes.
    """
    if isinstance(series, pd.Series):
        index = series.index
        arr = series.to_numpy()
    else:
        arr = np.asarray(series)
        index = None

    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    res = {f"fft_mag_{f}": np.full(n, np.nan, dtype=float) for f in freqs}
    freqs = list(freqs)

    for end in range(window, n + 1):
        segment = arr[end - window : end]
        fft_vals = np.abs(np.fft.rfft(segment))
        fft_freqs = np.fft.rfftfreq(window, d=1)
        for f in freqs:
            idx = np.argmin(np.abs(fft_freqs - f))
            res[f"fft_mag_{f}"][end - 1] = fft_vals[idx]

    return pd.DataFrame(res, index=index)


def rolling_wavelet_features(
    series: pd.Series | np.ndarray,
    window: int = 128,
    wavelet: str = "db4",
    level: int = 2,
) -> pd.DataFrame:
    """Compute wavelet detail coefficients over rolling windows.

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Input signal.
    window : int, default 128
        Rolling window size.
    wavelet : str, default "db4"
        Wavelet name passed to :func:`pywt.wavedec`.
    level : int, default 2
        Decomposition level.

    Returns
    -------
    pd.DataFrame
        DataFrame with column ``wavelet_<wavelet>_lvl<level>`` containing the
        mean absolute detail coefficient at the specified level.
    """
    if isinstance(series, pd.Series):
        index = series.index
        arr = series.to_numpy()
    else:
        arr = np.asarray(series)
        index = None

    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    name = f"wavelet_{wavelet}_lvl{level}"
    res = np.full(n, np.nan, dtype=float)

    for end in range(window, n + 1):
        segment = arr[end - window : end]
        coeffs = pywt.wavedec(segment, wavelet, level=level)
        detail = coeffs[level]
        res[end - 1] = np.mean(np.abs(detail))

    return pd.DataFrame({name: res}, index=index)


def spectral_features(
    series: pd.Series | np.ndarray, window: int = 128
) -> pd.DataFrame:
    """Compute rolling spectral energy using FFT.

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Input signal.
    window : int, default 128
        Rolling window size.

    Returns
    -------
    pd.DataFrame
        DataFrame with column ``spec_energy`` containing the spectral
        energy of each window.
    """
    if isinstance(series, pd.Series):
        index = series.index
        arr = series.to_numpy(dtype=float)
    else:
        arr = np.asarray(series, dtype=float)
        index = None

    n = len(arr)
    energy = np.full(n, np.nan, dtype=float)

    for end in range(window, n + 1):
        segment = arr[end - window : end]
        fft_vals = np.fft.rfft(segment)
        energy[end - 1] = np.sum(np.abs(fft_vals) ** 2)

    return pd.DataFrame({"spec_energy": energy}, index=index)


def wavelet_energy(
    series: pd.Series | np.ndarray,
    window: int = 128,
    wavelet: str = "db4",
    level: int = 2,
) -> pd.DataFrame:
    """Compute rolling energy of wavelet detail coefficients.

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Input signal.
    window : int, default 128
        Rolling window size.
    wavelet : str, default ``"db4"``
        Wavelet name passed to :func:`pywt.wavedec`.
    level : int, default 2
        Decomposition level.

    Returns
    -------
    pd.DataFrame
        DataFrame with column ``wavelet_energy`` containing the sum of
        squared detail coefficients at the specified level.
    """
    if isinstance(series, pd.Series):
        index = series.index
        arr = series.to_numpy(dtype=float)
    else:
        arr = np.asarray(series, dtype=float)
        index = None

    n = len(arr)
    energy = np.full(n, np.nan, dtype=float)

    for end in range(window, n + 1):
        segment = arr[end - window : end]
        coeffs = pywt.wavedec(segment, wavelet, level=level)
        detail = coeffs[level]
        energy[end - 1] = np.sum(np.square(detail))

    return pd.DataFrame({"wavelet_energy": energy}, index=index)
