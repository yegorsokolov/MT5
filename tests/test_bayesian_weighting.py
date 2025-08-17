import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from strategy.bayesian_weighting import BayesianWeighting


def test_posterior_updates():
    bw = BayesianWeighting(["algo1"])
    prior = bw.posterior("algo1")
    bw.log_pnl("algo1", 0.1)
    updated = bw.posterior("algo1")
    assert updated["mu"] != prior["mu"]
    assert updated["lambda"] > prior["lambda"]


def test_conservative_allocation_high_uncertainty():
    bw = BayesianWeighting(["stable", "new"])
    for _ in range(50):
        bw.log_pnl("stable", 0.01)
    bw.log_pnl("new", 0.05)
    weights = bw.weights()
    assert weights["stable"] > weights["new"]
    assert weights["new"] < 0.2
