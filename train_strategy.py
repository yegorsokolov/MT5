"""Training entry point for strategy search models.

The script exposes a very small command line interface used in the tests.  When
``--graph-search`` is supplied a :class:`models.strategy_search.StrategySearchNet`
is trained using policy gradients on a toy dataset and the resulting strategy is
evaluated using the existing execution engine from
``strategies.graph_dsl.StrategyGraph``.

This module is intentionally minimal; the goal is simply to demonstrate how the
pieces fit together rather than provide an industrial strength training
pipeline.
"""

from __future__ import annotations

import argparse

import torch

from models.strategy_search import train_strategy_search


def train_strategy(data, graph_search: bool = False, episodes: int = 100):
    """Helper used by the unit tests to trigger training."""

    if graph_search:
        return train_strategy_search(data, episodes=episodes)
    raise ValueError("only graph_search training is implemented in this demo")


def main() -> None:  # pragma: no cover - exercised via subprocess in tests
    parser = argparse.ArgumentParser(description="Train trading strategies")
    parser.add_argument("--graph-search", action="store_true", help="train StrategySearchNet")
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
        model = train_strategy_search(data, episodes=args.episodes)
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

