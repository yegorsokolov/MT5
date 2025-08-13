import numpy as np

from rl.multi_objective import weighted_sum, pareto_frontier


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
