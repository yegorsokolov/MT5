"""Policy network for dynamic capital allocation.

This lightweight network observes per-strategy profit and risk metrics and
produces allocation weights that sum to one.  It is intentionally simple so it
can run in environments without deep learning frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    """Return softmax of ``x`` in a numerically stable manner."""
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


@dataclass
class CapitalAllocator:
    """Allocate capital between strategies using a tiny policy network.

    The policy computes a score for each strategy as ``alpha * pnl - beta * risk``
    where ``pnl`` and ``risk`` are per-strategy metrics.  Scores are normalised
    with a softmax so resulting weights sum to one.
    """

    alpha: float = 1.0
    beta: float = 1.0
    strategies: Sequence[str] | None = None
    _last_allocation: Dict[str, float] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------
    def allocate(self, pnl: Dict[str, float], risk: Dict[str, float]) -> Dict[str, float]:
        """Return normalised allocation weights for the provided metrics."""
        keys = sorted(pnl.keys())
        pnl_arr = np.array([pnl[k] for k in keys], dtype=float)
        risk_arr = np.array([risk.get(k, 0.0) for k in keys], dtype=float)
        scores = self.alpha * pnl_arr - self.beta * risk_arr
        weights = _softmax(scores)
        self._last_allocation = {k: float(w) for k, w in zip(keys, weights)}
        return self._last_allocation

    # ------------------------------------------------------------------
    def train(
        self,
        pnl: np.ndarray,
        risk: np.ndarray,
        target: np.ndarray,
        lr: float = 0.01,
        epochs: int = 500,
    ) -> None:
        """Fit ``alpha`` and ``beta`` to match ``target`` allocations."""
        pnl = np.asarray(pnl, dtype=float)
        risk = np.asarray(risk, dtype=float)
        target = np.asarray(target, dtype=float)
        for _ in range(epochs):
            scores = self.alpha * pnl - self.beta * risk
            pred = np.exp(scores - scores.max(axis=1, keepdims=True))
            pred = pred / pred.sum(axis=1, keepdims=True)
            err = pred - target
            grad_alpha = np.sum(err * pnl)
            grad_beta = -np.sum(err * risk)
            self.alpha -= lr * grad_alpha
            self.beta -= lr * grad_beta
