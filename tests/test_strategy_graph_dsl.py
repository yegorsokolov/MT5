import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root on path for direct execution
sys.path.append(str(Path(__file__).resolve().parents[1]))

from strategies.graph_dsl import (
    Indicator,
    Filter,
    PositionSizer,
    ExitRule,
    StrategyGraph,
)
from models.strategy_graph_controller import StrategyGraphController


def test_graph_execution():
    nodes = {
        0: Indicator("price", ">", "ma"),
        1: Filter(),
        2: PositionSizer(1.0),
        3: ExitRule(),
    }
    edges = [(0, 1, None), (1, 2, True), (1, 3, False)]
    graph = StrategyGraph(nodes=nodes, edges=edges)
    data = [
        {"price": 1.0, "ma": 0.0},
        {"price": 2.0, "ma": 3.0},
    ]
    pnl = graph.run(data)
    assert pnl == pytest.approx(1.0)


def test_controller_generation():
    controller = StrategyGraphController(input_dim=2)
    features = np.array([[1.0, 2.0]])
    graph = controller.generate(features, risk=0.8)
    assert isinstance(graph, StrategyGraph)
    pnl = graph.run(
        [
            {"price": 1.0, "ma": 0.0},
            {"price": 2.0, "ma": 3.0},
        ]
    )
    assert isinstance(pnl, float)
