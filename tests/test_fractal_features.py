import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.fractal_features import (
    rolling_fractal_dimension,
    rolling_hurst_exponent,
)


def test_trend_series_has_high_hurst_low_dimension():
    n = 200
    series = pd.Series(np.linspace(0, 1, n))
    hurst = rolling_hurst_exponent(series, window=n).iloc[-1]
    fd = rolling_fractal_dimension(series, window=n).iloc[-1]
    assert hurst > 0.9
    assert fd < 1.1


def test_random_walk_series_hurst_near_half_dimension_high():
    np.random.seed(0)
    n = 400
    series = pd.Series(np.cumsum(np.random.randn(n)))
    hurst = rolling_hurst_exponent(series, window=n).iloc[-1]
    fd = rolling_fractal_dimension(series, window=n).iloc[-1]
    assert 0.4 < hurst < 0.7
    assert 1.3 < fd < 2.0
