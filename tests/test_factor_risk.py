import numpy as np
import pandas as pd

from portfolio.factor_risk import FactorRisk
from risk_manager import RiskManager


def test_factor_exposure_regression():
    np.random.seed(0)
    n = 200
    factors = pd.DataFrame({
        "carry": np.random.randn(n),
        "momentum": np.random.randn(n),
    })
    true_exp = np.array([0.5, 2.0])
    noise = np.random.randn(n) * 0.01
    asset_returns = factors.values @ true_exp + noise
    fr = FactorRisk(factors)
    exposures = fr.compute_exposures(asset_returns)
    assert np.allclose(exposures["carry"], true_exp[0], atol=0.1)
    assert np.allclose(exposures["momentum"], true_exp[1], atol=0.1)


def test_risk_manager_tracks_factor_contributions():
    np.random.seed(1)
    n = 80
    factors = pd.DataFrame({
        "carry": np.random.randn(n),
        "momentum": np.random.randn(n),
    })
    # set final step to ones so contributions equal exposures
    factors.iloc[-1] = [1.0, 1.0]
    true_exp = np.array([1.5, -0.7])
    asset_returns = factors.values @ true_exp + np.random.randn(n) * 0.01
    rm = RiskManager(max_drawdown=1e9, var_window=n, initial_capital=1.0)
    for i in range(n):
        fr_dict = {
            "carry": float(factors.iloc[i, 0]),
            "momentum": float(factors.iloc[i, 1]),
        }
        rm.update("bot", pnl=float(asset_returns[i]), exposure=0.0, check_hedge=False, factor_returns=fr_dict)
    status = rm.status()
    contrib = status["factor_contributions"]
    assert np.allclose(contrib["carry"], true_exp[0], atol=0.1)
    assert np.allclose(contrib["momentum"], true_exp[1], atol=0.1)
