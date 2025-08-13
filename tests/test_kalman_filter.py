import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.kalman_filter import kalman_smooth


def test_kalman_smooth_reduces_noise():
    np.random.seed(0)
    n = 200
    true_price = np.cumsum(np.random.normal(0, 0.1, n))
    noisy_price = true_price + np.random.normal(0, 0.5, n)
    series = pd.Series(noisy_price)

    filtered = kalman_smooth(series)

    mae_noisy = np.mean(np.abs(noisy_price - true_price))
    mae_filtered = np.mean(np.abs(filtered["price"].values - true_price))

    assert mae_filtered < mae_noisy
    assert (filtered["volatility"] >= 0).all()
