import time

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.garch_vol import garch_volatility
from analysis.frequency_features import (
    rolling_fft_features,
    rolling_wavelet_features,
)
from analysis.fractal_features import rolling_fractal_features

try:
    from analysis import gpu_features as gf
except Exception:  # pragma: no cover - GPU optional
    gf = None  # type: ignore

from utils.resource_monitor import monitor

HAVE_GPU = bool(
    gf
    and gf.cp is not None
    and getattr(monitor.capabilities, "gpu", getattr(monitor.capabilities, "has_gpu", False))
)


@pytest.mark.skipif(not HAVE_GPU, reason="GPU or CuPy not available")
def test_gpu_cpu_parity():
    rng = np.random.default_rng(0)
    data = pd.Series(rng.standard_normal(512))
    cpu_vol = garch_volatility(data)
    gpu_vol = gf.garch_volatility_gpu(data)
    assert np.allclose(cpu_vol.fillna(0).values, gpu_vol.fillna(0).values, atol=1e-6)

    cpu_fft = rolling_fft_features(data)
    gpu_fft = gf.rolling_fft_features_gpu(data)
    assert np.allclose(
        cpu_fft.fillna(0).values, gpu_fft.fillna(0).values, atol=1e-6
    )

    cpu_wave = rolling_wavelet_features(data)
    gpu_wave = gf.rolling_wavelet_features_gpu(data)
    assert np.allclose(
        cpu_wave.fillna(0).values, gpu_wave.fillna(0).values, atol=1e-6
    )

    cpu_frac = rolling_fractal_features(data)
    gpu_frac = gf.rolling_fractal_features_gpu(data)
    assert np.allclose(
        cpu_frac.fillna(0).values, gpu_frac.fillna(0).values, atol=1e-6
    )


@pytest.mark.skipif(not HAVE_GPU, reason="GPU or CuPy not available")
def test_gpu_speed_benchmark():
    rng = np.random.default_rng(0)
    data = pd.Series(rng.standard_normal(4096))

    start = time.time()
    rolling_fft_features(data)
    rolling_wavelet_features(data)
    rolling_fractal_features(data)
    cpu_time = time.time() - start

    start = time.time()
    gf.rolling_fft_features_gpu(data)
    gf.rolling_wavelet_features_gpu(data)
    gf.rolling_fractal_features_gpu(data)
    gpu_time = time.time() - start

    # Allow some tolerance; GPU should not be significantly slower
    assert gpu_time <= cpu_time * 1.5
