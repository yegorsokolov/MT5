import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pywt
sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.frequency_features import (
    rolling_fft_features,
    rolling_wavelet_features,
)


def test_rolling_fft_detects_frequency():
    n = 256
    t = np.arange(n)
    freq = 0.01
    series = pd.Series(np.sin(2 * np.pi * freq * t))
    res = rolling_fft_features(series, window=128, freqs=[freq])
    last_val = res[f"fft_mag_{freq}"].iloc[-1]
    fft_vals = np.abs(np.fft.rfft(series.to_numpy()[-128:]))
    fft_freqs = np.fft.rfftfreq(128, d=1)
    expected = fft_vals[np.argmin(np.abs(fft_freqs - freq))]
    assert np.isclose(last_val, expected, atol=1e-6)


def test_rolling_wavelet_matches_pywt():
    n = 128
    t = np.arange(n)
    series = pd.Series(np.sin(2 * np.pi * 0.05 * t))
    res = rolling_wavelet_features(series, window=n, wavelet="db4", level=2)
    coeffs = pywt.wavedec(series.to_numpy(), "db4", level=2)
    expected = np.mean(np.abs(coeffs[2]))
    assert np.isclose(res["wavelet_db4_lvl2"].iloc[-1], expected, atol=1e-6)


