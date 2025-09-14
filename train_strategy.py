"""Training entry point for strategy search models.

The script exposes a very small command line interface used in the tests.  When
``--graph-search`` is supplied a :class:`models.strategy_graph_controller.StrategyGraphController`
is trained using policy gradients on a toy dataset and the resulting strategy is
evaluated using the execution engine from ``strategies.graph_dsl.StrategyGraph``.
The controller emits small graphs composed of :class:`strategies.graph_dsl.Indicator`,
``Filter`` and ``PositionSizer`` nodes which are executed to obtain a PnL reward.

This module is intentionally minimal; the goal is simply to demonstrate how the
pieces fit together rather than provide an industrial strength training
pipeline.
"""

from __future__ import annotations

import argparse
import asyncio

import torch

from models.strategy_graph_controller import train_strategy_graph_controller
from strategy.evolution_lab import EvolutionLab
from training.curriculum import build_live_strategy_curriculum


def train_strategy(data, graph_search: bool = False, episodes: int = 100):
    """Helper used by the unit tests to trigger training."""

    if graph_search:
        return train_strategy_graph_controller(data, episodes=episodes)
    raise ValueError("only graph_search training is implemented in this demo")


def evolve_with_curriculum(base_strategy, live_messages, seeds=(0, 1)):
    """Evolve multiple strategies and evaluate them on live data.

    The helper wires together :class:`strategy.evolution_lab.EvolutionLab` and
    :func:`training.curriculum.build_live_strategy_curriculum` so that unit
    tests can verify queue based evolution and curriculum progression.
    """

    lab = EvolutionLab(base_strategy)
    asyncio.run(lab.evolve_queue(seeds, live_messages))
    # Build a curriculum purely as an example – the returned scheduler is not
    # used further in this demonstration script but ensures the call path is
    # exercised by tests.
    def single():
        return lambda msg: float(msg.get("price", 0))

    def multi():
        return lambda msg: float(msg.get("price", 0) - msg.get("ma", 0))

    def graph():  # pragma: no cover - placeholder for complex graph strategies
        return lambda msg: float(msg.get("price", 0))

    build_live_strategy_curriculum(single, multi, graph, live_messages)


def main() -> None:  # pragma: no cover - exercised via subprocess in tests
    parser = argparse.ArgumentParser(description="Train trading strategies")
    parser.add_argument("--graph-search", action="store_true", help="train StrategyGraphController")
    parser.add_argument("--episodes", type=int, default=100, help="training episodes")
    args = parser.parse_args()

    # Synthetic data used purely for demonstration purposes
    data = [
        {"price": 1.0, "ma": 0.9},
        {"price": 1.1, "ma": 1.0},
        {"price": 1.2, "ma": 1.1},
        {"price": 1.3, "ma": 1.2},
    ]

    if args.graph_search:
        model = train_strategy_graph_controller(data, episodes=args.episodes)
        x = torch.zeros((2, 1))
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        logits = model(x, edge_index)
        action = int(torch.argmax(logits).item())
        graph = model.build_graph(action)
        pnl = graph.run(data)
        print(f"Trained strategy PnL: {pnl:.4f}")
    else:
        raise SystemExit("No training mode selected")


if __name__ == "__main__":
    main()

