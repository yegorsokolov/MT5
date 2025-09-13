import sys
from pathlib import Path

import pytest
import torch

# Ensure project root on path for direct execution
sys.path.append(str(Path(__file__).resolve().parents[1]))

from strategy_dsl import IndicatorNode, PositionNode, RiskNode, StrategyGraph
from models.strategy_graph_controller import (
    StrategyGraphController,
    train_strategy_graph_controller,
)


def build_sample_data():
    return [
        {"price": 1.0, "ma": 0.9},
        {"price": 1.1, "ma": 1.0},
        {"price": 1.2, "ma": 1.1},
        {"price": 1.3, "ma": 1.2},
    ]


def test_graph_execution_and_serialisation():
    nodes = {0: IndicatorNode("price", ">", "ma"), 1: RiskNode(0.5), 2: PositionNode(1.0)}
    edges = [(0, 1, None), (1, 2, True)]
    graph = StrategyGraph(nodes=nodes, edges=edges)
    data = build_sample_data()[:2]
    pnl = graph.run(data)
    assert pnl > 0
    payload = graph.to_dict()
    rebuilt = StrategyGraph.from_dict(payload)
    assert rebuilt.run(data) == pytest.approx(pnl)


def test_policy_gradient_training():
    data = build_sample_data()
    model = train_strategy_graph_controller(data, episodes=200, lr=0.1, seed=0)
    x = torch.zeros((2, 1))
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    logits = model(x, edge_index)
    probs = torch.softmax(logits, dim=-1)
    assert float(probs[0]) > 0.8  # model should strongly prefer action 0
    action = int(torch.argmax(probs).item())
    graph = model.build_graph(action)
    pnl = graph.run(data)
    assert pnl > 0

