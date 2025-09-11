import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.scenario_mc import generate_correlated_shocks
from risk_manager import RiskManager


def test_generate_correlated_shocks_shape():
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    shocks = generate_correlated_shocks(cov, n_steps=10, n_paths=3, df=4, rng=np.random.default_rng(0))
    assert shocks.shape == (3, 10, 2)


def test_extreme_loss_triggers_halt(monkeypatch):
    cov = np.array([[1.0]])
    rm = RiskManager(max_drawdown=0.05)
    metrics: list[tuple[str, float]] = []
    monkeypatch.setattr(
        "risk_manager.record_metric",
        lambda name, value, tags=None: metrics.append((name, value)),
    )
    rng = np.random.default_rng(0)
    rm.run_scenarios(cov, n_steps=5, n_paths=1, df=2, rng=rng)
    assert rm.metrics.trading_halted is True
    assert any(name == "scenario_mc_max_drawdown" for name, _ in metrics)
