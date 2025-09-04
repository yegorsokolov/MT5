from __future__ import annotations

"""Fuzzy consensus scoring for algorithm outputs.

This module provides a lightweight mechanism to gauge the level of agreement
between multiple trading signals.  Each algorithm produces an action in the
range ``[-1, 1]`` representing sell to buy conviction.  The consensus score is
computed using weighted fuzzy voting where ``1`` indicates perfect agreement and
``0`` represents complete conflict (e.g. equal buy and sell signals).
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


@dataclass
class FuzzyConsensus:
    """Compute agreement among algorithm signals using fuzzy logic."""

    weights: Dict[str, float] = field(default_factory=dict)

    def score(self, signals: Dict[str, float]) -> Tuple[float, float]:
        """Return the consensus score and weighted mean action.

        Parameters
        ----------
        signals:
            Mapping of algorithm name to its proposed action in ``[-1, 1]``.

        Returns
        -------
        Tuple[float, float]
            ``(consensus, mean_action)`` where ``consensus`` is in ``[0, 1]``.
        """

        if not signals:
            return 0.0, 0.0

        names = list(signals)
        weights = np.array([self.weights.get(n, 1.0) for n in names], dtype=float)
        values = np.clip(np.array([signals[n] for n in names], dtype=float), -1.0, 1.0)

        mean = float(np.average(values, weights=weights))
        deviation = float(np.average(np.abs(values - mean), weights=weights))
        consensus = float(np.clip(1.0 - deviation, 0.0, 1.0))
        return consensus, mean


__all__ = ["FuzzyConsensus"]
