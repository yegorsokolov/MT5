from __future__ import annotations

import numpy as np
from typing import List

from strategy_dsl import Buy, Sell, Wait, StrategyInterpreter

TOKENS = [Buy, Sell, Wait]


class StrategyController:
    """Simple logistic controller that predicts DSL action tokens."""

    def __init__(self, input_dim: int):
        rng = np.random.default_rng(0)
        self.w = rng.normal(scale=0.1, size=(input_dim, len(TOKENS)))
        self.b = np.zeros(len(TOKENS))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w + self.b

    def act(self, seq: np.ndarray) -> List[object]:
        logits = self.forward(seq)
        actions = logits.argmax(axis=1)
        mapping = {0: Buy(), 1: Sell(), 2: Wait()}
        return [mapping[i] for i in actions]


def _prepare_arrays(market_data: List[dict]) -> tuple[np.ndarray, np.ndarray]:
    features = np.array([[bar["price"], bar["ma"]] for bar in market_data], dtype=float)
    targets = []
    for bar in market_data:
        if bar["price"] < bar["ma"]:
            targets.append(0)
        elif bar["price"] > bar["ma"]:
            targets.append(1)
        else:
            targets.append(2)
    return features, np.array(targets)


def train_strategy_controller(
    market_data: List[dict] | None = None, epochs: int = 200, lr: float = 0.1
) -> StrategyController:
    if market_data is None:
        market_data = [
            {"price": 1.0, "ma": 2.0},
            {"price": 3.0, "ma": 2.0},
            {"price": 1.0, "ma": 2.0},
        ]
    features, targets = _prepare_arrays(market_data)
    controller = StrategyController(input_dim=features.shape[1])
    num_classes = len(TOKENS)
    for _ in range(epochs):
        logits = controller.forward(features)
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        one_hot = np.eye(num_classes)[targets]
        grad = (probs - one_hot) / len(features)
        controller.w -= lr * features.T @ grad
        controller.b -= lr * grad.sum(axis=0)
    return controller


def evaluate_controller(controller: StrategyController, market_data: List[dict]) -> float:
    features, _ = _prepare_arrays(market_data)
    actions = controller.act(features)
    interp = StrategyInterpreter()
    return interp.run(market_data, actions)
