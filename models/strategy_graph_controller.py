from __future__ import annotations

import numpy as np

from strategies.graph_dsl import (
    ExitRule,
    Filter,
    Indicator,
    PositionSizer,
    StrategyGraph,
)


class StrategyGraphController:
    """Generates simple strategy graphs from features and a risk profile."""

    def __init__(self, input_dim: int):
        rng = np.random.default_rng(0)
        self.w = rng.normal(scale=0.1, size=(input_dim,))

    def generate(self, features: np.ndarray, risk: float) -> StrategyGraph:
        score = float(features.mean() * self.w.mean())
        op = ">" if score >= 0 else "<"
        indicator = Indicator("price", op, "ma")
        size = 1.0 if risk >= 0.5 else 0.5
        nodes = {
            0: indicator,
            1: Filter(),
            2: PositionSizer(size),
            3: ExitRule(),
        }
        edges = [(0, 1, None), (1, 2, True), (1, 3, False)]
        return StrategyGraph(nodes=nodes, edges=edges)
