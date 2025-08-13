"""Utilities for multi-objective reinforcement learning.

Provides helpers to work with vector rewards and scalarisation
techniques such as weighted sums and Pareto frontiers.  The
implementations are intentionally lightweight to avoid heavy
dependencies while remaining useful for experimentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class VectorReward:
    """Simple container holding a vector of reward components."""

    values: np.ndarray

    def weighted_sum(self, weights: Sequence[float]) -> float:
        """Return a weighted sum of the reward components."""
        w = np.asarray(weights, dtype=np.float32)
        v = np.asarray(self.values, dtype=np.float32)
        if w.shape != v.shape:
            raise ValueError("weights and reward vector must have same length")
        return float(np.dot(w, v))


def weighted_sum(reward: Sequence[float], weights: Sequence[float]) -> float:
    """Compute a weighted sum for ``reward`` using ``weights``."""
    return VectorReward(np.asarray(reward, dtype=np.float32)).weighted_sum(weights)


def pareto_frontier(points: Iterable[Sequence[float]]) -> np.ndarray:
    """Return the Pareto optimal subset of ``points``.

    Parameters
    ----------
    points:
        Iterable of reward vectors.  Each point is assumed to represent an
        objective vector where higher is better for all components.
    """

    pts = np.asarray(list(points), dtype=np.float32)
    if pts.size == 0:
        return pts
    mask = np.ones(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        if not mask[i]:
            continue
        dominated = np.all(pts >= p, axis=1) & np.any(pts > p, axis=1)
        mask[dominated] = False
    return pts[mask]


__all__ = ["VectorReward", "weighted_sum", "pareto_frontier"]
