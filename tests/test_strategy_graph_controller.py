import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# Ensure project root on path for direct execution
sys.path.append(str(Path(__file__).resolve().parents[1]))

from strategies.graph_dsl import (
    ExitRule,
    Filter,
    Indicator,
    PositionSizer,
    StrategyGraph,
)
from models.strategy_graph_controller import train_strategy_graph_controller


def build_sample_data():
    return [
        {"price": 1.0, "ma": 0.9},
        {"price": 1.1, "ma": 1.0},
        {"price": 1.2, "ma": 1.1},
        {"price": 1.4, "ma": 1.5},
    ]


def test_graph_execution_and_serialisation():
    nodes = {
        0: Indicator("price", ">", "ma"),
        1: Filter(),
        2: PositionSizer(1.0),
        3: ExitRule(),
    }
    edges = [(0, 1, None), (1, 2, True), (1, 3, False)]
    graph = StrategyGraph(nodes=nodes, edges=edges)
    data = build_sample_data()
    pnl = graph.run(data)
    assert pnl > 0
    payload = graph.to_dict()
    rebuilt = StrategyGraph.from_dict(payload)
    assert rebuilt.run(data) == pytest.approx(pnl)


def test_policy_gradient_training():
    data = build_sample_data()
    model = train_strategy_graph_controller(data, episodes=200, lr=0.1, seed=0)
    x, summary = model.prepare_graph_inputs(data)
    logits = model(x)
    probs = torch.softmax(logits, dim=-1)
    assert float(probs[0]) > 0.8  # model should strongly prefer action 0
    action = int(torch.argmax(probs).item())
    macro = summary["macro"] if summary.get("has_macro") else None
    risk = max(0.0, min(1.0, 0.5 + 0.25 * abs(summary.get("signal", 0.0))))
    graph = model.build_graph(action, risk=risk, macro=macro)
    pnl = graph.run(data)
    assert pnl > 0


def test_generate_respects_features_and_risk():
    data = build_sample_data()
    model = train_strategy_graph_controller(data, episodes=200, lr=0.1, seed=0)
    bullish = [
        {"price": 1.5, "ma": 1.0},
        {"price": 1.6, "ma": 1.2},
    ]
    bearish = [
        {"price": 0.9, "ma": 1.1},
        {"price": 0.8, "ma": 1.0},
    ]
    long_graph, long_info = model.generate(bullish, risk_profile=0.9, return_info=True)
    short_graph, short_info = model.generate(bearish, risk_profile=0.1, return_info=True)

    assert isinstance(long_graph.nodes[0], Indicator)
    assert long_graph.nodes[0].op == ">"
    assert isinstance(short_graph.nodes[0], Indicator)
    assert short_graph.nodes[0].op == "<"
    assert long_graph.nodes[2].size > short_graph.nodes[2].size
    assert long_info["summary"]["signal"] > 0
    assert short_info["summary"]["signal"] < 0


def test_graph_search_cli_executes():
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(root / "mt5" / "train_strategy.py"),
        "--graph-search",
        "--episodes",
        "200",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = [line for line in result.stdout.splitlines() if "Trained strategy PnL" in line]
    assert lines, f"Unexpected CLI output: {result.stdout!r}"
    value = float(lines[0].split(":", 1)[1])
    assert value > 0.0

