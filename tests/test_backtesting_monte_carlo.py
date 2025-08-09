import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtesting import monte_carlo


def test_bootstrap_and_stability():
    stable = pd.Series([0.01] * 100)
    unstable = pd.Series([0.1, -0.1] * 50)

    dist = monte_carlo.bootstrap_sharpe(stable, n_iterations=50, seed=1)
    assert len(dist) == 50
    stats = monte_carlo.analyze_variance(stable, n_iterations=50, seed=1)
    assert set(stats) == {"sharpe_mean", "sharpe_std"}

    assert monte_carlo.is_stable(stable, threshold=0.05, n_iterations=50, seed=1)
    assert not monte_carlo.is_stable(unstable, threshold=0.05, n_iterations=50, seed=1)
