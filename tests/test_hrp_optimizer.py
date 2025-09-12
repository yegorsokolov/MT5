import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from portfolio import HRPOptimizer, PortfolioOptimizer


def test_hrp_vs_mean_variance_diversification():
    # Covariance with two highly correlated clusters
    cov = np.array([
        [0.1, 0.09, 0.01, 0.01],
        [0.09, 0.1, 0.01, 0.01],
        [0.01, 0.01, 0.1, 0.09],
        [0.01, 0.01, 0.09, 0.1],
    ])
    mu = np.array([0.08, 0.09, 0.05, 0.04])
    mv = PortfolioOptimizer()
    w_mv = mv.compute_weights(mu, cov)
    hrp = HRPOptimizer()
    w_hrp = hrp.compute_weights(mu, cov)
    assert np.isclose(w_hrp.sum(), 1.0)
    assert np.isclose(np.sum(np.abs(w_mv)), 1.0)
    assert hrp.diversification_ratio() >= mv.diversification_ratio()
