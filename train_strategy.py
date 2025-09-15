"""Training entry point for strategy search models and registry demos.

The script exposes two small command line utilities used in tests:

* ``--graph-search`` trains a
  :class:`models.strategy_graph_controller.StrategyGraphController` using policy
  gradients on a toy dataset and prints the resulting PnL.
* ``--strategy`` executes a registered strategy from :mod:`strategies` on the
  same synthetic dataset, demonstrating the dynamic registry and plugin system.

The controller emits small graphs composed of
:class:`strategies.graph_dsl.Indicator`, ``Filter`` and ``PositionSizer`` nodes
which are executed to obtain a PnL reward.  Strategies retrieved from the
registry simply implement ``generate_order`` and optional ``update`` methods.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
from typing import Dict, Iterable, List

import torch

from models.strategy_graph_controller import train_strategy_graph_controller
from strategies import create_strategy, iter_strategies
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


def _parse_params(params: Iterable[str]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for item in params:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"Invalid parameter override '{item}', expected key=value")
        try:
            result[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            result[key] = value
    return result


def run_registered_strategy(name: str, market_data: List[Dict[str, float]], **kwargs):
    """Execute ``name`` on ``market_data`` and return generated orders."""

    strategy = create_strategy(name, **kwargs)
    orders = []
    for tick in market_data:
        orders.append(strategy.generate_order(tick))
    return orders


def main() -> None:  # pragma: no cover - exercised via subprocess in tests
    parser = argparse.ArgumentParser(description="Train trading strategies")
    parser.add_argument("--graph-search", action="store_true", help="train StrategyGraphController")
    parser.add_argument("--episodes", type=int, default=100, help="training episodes")
    parser.add_argument("--strategy", type=str, help="run a registered strategy on demo data")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Strategy parameter override (can be specified multiple times)",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List strategies registered in the dynamic registry and exit",
    )
    args = parser.parse_args()

    if args.graph_search and args.strategy:
        raise SystemExit("Choose either --graph-search or --strategy")

    if args.list_strategies:
        for name, spec in iter_strategies():
            description = f" - {spec.description}" if spec.description else ""
            print(f"{name}{description}")
        return

    # Synthetic data used purely for demonstration purposes
    data = [
        {"price": 1.0, "ma": 0.9},
        {"price": 1.1, "ma": 1.0},
        {"price": 1.2, "ma": 1.1},
        {"price": 1.3, "ma": 1.2},
    ]

    if args.strategy:
        params = _parse_params(args.param)
        try:
            orders = run_registered_strategy(args.strategy, data, **params)
        except KeyError as exc:  # pragma: no cover - user error path
            available = ", ".join(sorted(name for name, _ in iter_strategies()))
            raise SystemExit(
                f"Unknown strategy {exc.args[0]!r}. Available strategies: {available}"
            ) from exc
        for tick, order in zip(data, orders):
            print(f"price={tick['price']:.2f} -> order={order}")
        return

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
