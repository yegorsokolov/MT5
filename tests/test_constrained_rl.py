import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "constrained_agent",
    Path(__file__).resolve().parents[1] / "rl" / "constrained_agent.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)  # type: ignore[arg-type]
ConstrainedAgent = module.ConstrainedAgent


def test_constrained_agent_respects_budget():
    agent = ConstrainedAgent(n_actions=2, risk_budget=0.2, lr=0.1)
    agent.update(1, 2.0, 0.5)
    for _ in range(200):
        action = agent.act(None)
        reward, cost = ((1.0, 0.1) if action == 0 else (2.0, 0.5))
        agent.update(action, reward, cost)
    avg_cost = agent.avg_cost(window=50)
    assert avg_cost <= 0.2 + 1e-6
    assert agent.lambda_ > 0
    actions = [agent.act(None) for _ in range(20)]
    assert sum(actions) <= 4
