import numpy as np
from prometheus_client import Gauge

import metrics
from portfolio.optimizer import PortfolioOptimizer, PortfolioRebalancer


def test_compute_weights():
    opt = PortfolioOptimizer()
    mu = np.array([0.1, 0.2])
    cov = np.array([[0.1, 0.02], [0.02, 0.08]])
    w = opt.compute_weights(mu, cov)
    assert np.isclose(w.sum(), 1.0)
    assert w[1] > w[0]


def test_rebalancer_updates_metrics(monkeypatch):
    dd = Gauge("dd", "")
    div = Gauge("div", "")
    monkeypatch.setattr(metrics, "PORTFOLIO_DRAWDOWN", dd)
    monkeypatch.setattr(metrics, "DIVERSIFICATION_RATIO", div)
    opt = PortfolioOptimizer()
    rb = PortfolioRebalancer(opt, rebalance_interval=1)
    mu = np.array([0.1, 0.2])
    cov = np.eye(2)
    weights = rb.update(mu, cov, portfolio_return=0.0)
    assert weights is not None
    assert div._value.get() > 0
