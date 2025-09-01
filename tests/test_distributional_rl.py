import importlib.util
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "distributional_agent", repo_root / "rl" / "distributional_agent.py"
)
dist_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dist_mod)
DistributionalAgent = dist_mod.DistributionalAgent
MeanAgent = dist_mod.MeanAgent


def test_distributional_risk_metrics():
    # returns with a heavy negative tail
    rewards = [0.02, -0.03, 0.015, -0.07, 0.04, -0.5, 0.03, 0.02, -0.01, 0.05]

    dist_agent = DistributionalAgent(n_actions=1, n_quantiles=5)
    mean_agent = MeanAgent(n_actions=1)

    for r in rewards:
        dist_agent.update(0, r)
        mean_agent.update(0, r)

    var_dist = dist_agent.value_at_risk(0, 0.1)
    var_mean = mean_agent.value_at_risk(0, 0.1)
    sharpe_dist = dist_agent.sharpe_ratio(0)
    sharpe_mean = mean_agent.sharpe_ratio(0)

    # distributional agent should recognise the heavy tail
    assert var_dist < var_mean
    # accounting for variance lowers the Sharpe ratio compared to the mean-only agent
    assert sharpe_dist < sharpe_mean
