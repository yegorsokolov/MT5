import os
import sys
import numpy as np

sys.path.append(os.getcwd())

from rl.multi_objective import weighted_sum, pareto_frontier
from analysis.multi_objective import TradeMetrics, weighted_sum as anal_weighted_sum


def test_weighted_sum_preference_shift():
    rewards = np.array([[1.0, -0.8], [0.5, -0.1]])
    w_ret = [1.0, 0.0]
    w_risk = [0.5, 0.5]
    pref_ret = np.argmax([weighted_sum(r, w_ret) for r in rewards])
    pref_risk = np.argmax([weighted_sum(r, w_risk) for r in rewards])
    assert pref_ret != pref_risk


def test_pareto_frontier():
    pts = np.array([[1.0, 0.0], [0.5, 0.5], [0.2, 0.1]])
    frontier = pareto_frontier(pts)
    # The last point is dominated by the first two
    assert frontier.shape[0] == 2


def test_profit_shift_with_weights():
    a = TradeMetrics(f1=0.9, expected_return=0.1, drawdown=0.05)
    b = TradeMetrics(f1=0.7, expected_return=0.3, drawdown=0.1)
    w_ret = {"f1": 0.0, "return": 1.0, "drawdown": 0.0}
    w_f1 = {"f1": 1.0, "return": 0.0, "drawdown": 0.0}
    pref_ret = np.argmax([anal_weighted_sum(m, w_ret) for m in (a, b)])
    pref_f1 = np.argmax([anal_weighted_sum(m, w_f1) for m in (a, b)])
    assert pref_ret == 1  # strategy b has higher return
    assert pref_f1 == 0  # strategy a has higher f1
