import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import analytics.metrics_store as ms
from portfolio.optimizer import PortfolioOptimizer, PortfolioRebalancer


def test_compute_weights():
    opt = PortfolioOptimizer()
    mu = np.array([0.1, 0.2])
    cov = np.array([[0.1, 0.02], [0.02, 0.08]])
    w = opt.compute_weights(mu, cov)
    assert np.isclose(w.sum(), 1.0)
    assert w[1] > w[0]


def test_rebalancer_updates_metrics(monkeypatch):
    calls = []
    def fake_record(name, value, tags=None):
        calls.append((name, value))
    monkeypatch.setattr(ms, "record_metric", fake_record)
    import portfolio.optimizer as po
    monkeypatch.setattr(po, "record_metric", fake_record)
    opt = PortfolioOptimizer()
    rb = PortfolioRebalancer(opt, rebalance_interval=1)
    mu = np.array([0.1, 0.2])
    cov = np.eye(2)
    weights = rb.update(mu, cov, portfolio_return=0.0)
    assert weights is not None
    assert any(c[0] == "diversification_ratio" for c in calls)
