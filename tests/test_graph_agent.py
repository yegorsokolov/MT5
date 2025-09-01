import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

# import build_rolling_adjacency
spec = importlib.util.spec_from_file_location(
    "graph_builder", Path(__file__).resolve().parents[1] / "data" / "graph_builder.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # type: ignore
build_rolling_adjacency = mod.build_rolling_adjacency

# import GraphAgent
spec_agent = importlib.util.spec_from_file_location(
    "graph_agent", Path(__file__).resolve().parents[1] / "rl" / "graph_agent.py"
)
agent_mod = importlib.util.module_from_spec(spec_agent)
spec_agent.loader.exec_module(agent_mod)  # type: ignore
GraphAgent = agent_mod.GraphAgent


def test_build_rolling_adjacency_detects_corr():
    ts = pd.date_range("2020-01-01", periods=20, freq="D")
    a = np.linspace(1, 2, 20)
    b = a * 2
    df = pd.DataFrame(
        {
            "Timestamp": list(ts) * 2,
            "Symbol": ["A"] * 20 + ["B"] * 20,
            "Price": np.concatenate([a, b]),
        }
    )
    mats = build_rolling_adjacency(df, window=5)
    assert mats, "Should produce adjacency matrices"
    last = list(mats.values())[-1]
    assert last.shape == (2, 2)
    assert last[0, 1] > 0.9


def test_graph_agent_policy_improves():
    adj = np.ones((2, 2))
    feats = np.array([[0.0], [1.0]])
    agent = GraphAgent(in_features=1, hidden_dim=8, num_actions=2)
    for _ in range(50):
        action = agent.act(feats, adj)
        reward = 1.0 if action == 1 else 0.0
        agent.store(feats, adj, action, reward)
        agent.train()
    assert agent.act(feats, adj) == 1
