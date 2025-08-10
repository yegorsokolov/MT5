import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.hierarchical_agent import HierarchicalAgent, ConstantPolicy


class SignManager:
    """Selects policy based on sign of observation."""

    def __init__(self):
        self.updated = False

    def select_policy(self, obs):
        return "trend" if float(np.mean(obs)) >= 0 else "mean"

    def update(self, batch):
        self.updated = True


def test_policy_selection():
    manager = SignManager()
    workers = {"trend": ConstantPolicy(1), "mean": ConstantPolicy(-1)}
    agent = HierarchicalAgent(manager, workers)
    act = agent.act(np.array([0.5]))
    assert act["manager"] == "trend"
    assert act["worker"] == 1
    act = agent.act(np.array([-0.1]))
    assert act["manager"] == "mean"
    assert act["worker"] == -1


def test_joint_learning():
    manager = SignManager()
    workers = {"trend": ConstantPolicy(1), "mean": ConstantPolicy(-1)}
    agent = HierarchicalAgent(manager, workers)
    # generate experience for both policies
    obs = np.array([0.2])
    act = agent.act(obs)
    agent.store(obs, act, 1.0, obs, False)
    obs2 = np.array([-0.2])
    act2 = agent.act(obs2)
    agent.store(obs2, act2, 1.0, obs2, False)
    agent.train(batch_size=2)
    assert workers["trend"].updates == 1
    assert workers["mean"].updates == 1
    assert manager.updated
