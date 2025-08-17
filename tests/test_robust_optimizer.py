import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from portfolio.robust_optimizer import RobustOptimizer


def test_robust_optimizer_stable_weights():
    rng = np.random.default_rng(0)
    mu = np.array([0.1, 0.12, 0.09])
    cov = np.array([[0.05, 0.01, 0.0], [0.01, 0.06, 0.02], [0.0, 0.02, 0.07]])
    opt = RobustOptimizer(ambiguity=0.5)
    w1 = opt.compute_weights(mu, cov)
    mu_p = mu + rng.normal(scale=0.01, size=3)
    cov_p = cov + rng.normal(scale=0.005, size=cov.shape)
    cov_p = (cov_p + cov_p.T) / 2
    w2 = opt.compute_weights(mu_p, cov_p)
    assert np.isclose(w1.sum(), 1.0)
    assert np.isclose(w2.sum(), 1.0)
    assert np.linalg.norm(w1 - w2) < 0.2


def test_risk_manager_rebalances_by_regime(monkeypatch):
    import importlib

    # prevent scheduler side effects before importing risk_manager
    monkeypatch.setitem(
        sys.modules,
        'scheduler',
        types.SimpleNamespace(start_scheduler=lambda: None),
    )
    rm_module = importlib.reload(importlib.import_module('risk_manager'))
    RiskManager = rm_module.RiskManager

    rm = RiskManager(max_drawdown=100, initial_capital=1.0)
    for _ in range(20):
        rm.update('s1', 0.01, factor_returns={'regime': 0})
        rm.update('s2', 0.02, factor_returns={'regime': 0})
    b1 = rm.rebalance_budgets(regime=0)
    for _ in range(20):
        rm.update('s1', 0.011, factor_returns={'regime': 0})
        rm.update('s2', 0.019, factor_returns={'regime': 0})
    b2 = rm.rebalance_budgets(regime=0)
    v1 = np.array([b1['s1'], b1['s2']])
    v2 = np.array([b2['s1'], b2['s2']])
    assert np.isclose(v1.sum(), rm.budget_allocator.capital)
    assert np.isclose(v2.sum(), rm.budget_allocator.capital)
    assert np.linalg.norm(v1 - v2) / rm.budget_allocator.capital < 0.3
