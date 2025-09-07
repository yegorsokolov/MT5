import numpy as np
import sys
from pathlib import Path

# Ensure repository root on path for direct module imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.hierarchical_agent import (
    HierarchicalAgent,
    EpsilonGreedyManager,
    TrendPolicy,
    MeanReversionPolicy,
    NewsDrivenPolicy,
)


def test_policy_selection():
    """Manager should choose policy with highest value estimate."""

    policies = ["mean_reversion", "news", "trend"]
    manager = EpsilonGreedyManager(policies, epsilon=0.0)
    manager.q_values.update({"trend": 1.0, "mean_reversion": 0.5, "news": -0.2})
    workers = {
        "mean_reversion": MeanReversionPolicy(),
        "news": NewsDrivenPolicy(),
        "trend": TrendPolicy(),
    }
    agent = HierarchicalAgent(manager, workers)
    act = agent.act(np.array([0.0]))
    # trend policy should be chosen -> mapped to its index
    assert act["manager"] == agent.policy_to_idx["trend"]
    assert act["worker"] == 1.0


def test_joint_learning():
    """Manager and all workers should be updated from shared buffer."""

    workers = {
        "mean_reversion": MeanReversionPolicy(),
        "news": NewsDrivenPolicy(),
        "trend": TrendPolicy(),
    }
    manager = EpsilonGreedyManager(list(workers.keys()), epsilon=0.0)
    agent = HierarchicalAgent(manager, workers)

    # Force policy sequence so each worker is selected once
    seq = iter(["trend", "mean_reversion", "news"])
    manager.select_policy = lambda obs, it=seq: next(it)

    obs = np.array([0.0])
    for _ in range(3):
        act = agent.act(obs)
        agent.store(obs, act, 1.0, obs, False)

    agent.train(batch_size=3)

    assert manager.updated
    for w in workers.values():
        assert w.updates == 1
