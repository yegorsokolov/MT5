import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.garch_vol import garch_volatility


def simulate_garch(n=500, omega=0.1, alpha=0.1, beta=0.8):
    """Generate a simple GARCH(1,1) return series."""
    eps = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * np.random.randn()
    returns = pd.Series(eps)
    true_vol = pd.Series(np.sqrt(sigma2))
    return returns, true_vol


def test_garch_volatility_tracks_true_vol():
    returns, true_vol = simulate_garch()
    est_vol = garch_volatility(returns)
    corr = est_vol.corr(true_vol)
    assert corr > 0.4
