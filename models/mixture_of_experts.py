"""Mixture-of-experts models with a gating network.

The module defines simple expert models and a gating network that assigns
probabilities to each expert based on the current market regime and the
available system resources.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

# ``utils.resource_monitor`` pulls in optional heavy deps in its package
# ``__init__``. Import it directly from the file path to keep tests light.
_spec = importlib.util.spec_from_file_location(
    "resource_monitor",
    Path(__file__).resolve().parents[1] / "utils" / "resource_monitor.py",
)
_rm = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_rm)  # type: ignore

ResourceCapabilities = _rm.ResourceCapabilities


class Expert:
    """Base class for expert models."""

    def predict(self, history: Sequence[float]) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class TrendExpert(Expert):
    """Predicts continuation of the latest trend."""

    def predict(self, history: Sequence[float]) -> float:
        if len(history) < 2:
            return history[-1]
        return history[-1] + (history[-1] - history[-2])


class MeanReversionExpert(Expert):
    """Predicts a move back toward a long-run mean of zero."""

    def predict(self, history: Sequence[float]) -> float:
        return 0.5 * history[-1]


class MacroExpert(Expert):
    """A simple macro fundamental based expert returning zero."""

    def predict(self, history: Sequence[float]) -> float:
        return 0.0


@dataclass
class ExpertSpec:
    """Expert model and its resource requirements."""

    model: Expert
    requirements: ResourceCapabilities


@dataclass
class GatingNetwork:
    """Soft gating network selecting experts based on regime and resources."""

    experts: Sequence[ExpertSpec]
    sharpness: float = 5.0

    def _supported(self, caps: ResourceCapabilities, req: ResourceCapabilities) -> bool:
        return (
            caps.cpus >= req.cpus
            and caps.memory_gb >= req.memory_gb
            and (not req.has_gpu or caps.has_gpu)
        )

    def weights(self, regime: float, caps: ResourceCapabilities) -> np.ndarray:
        """Return softmax weights over experts."""

        scores = []
        for idx, spec in enumerate(self.experts):
            if not self._supported(caps, spec.requirements):
                scores.append(-np.inf)
            else:
                scores.append(-self.sharpness * abs(regime - idx))
        scores_arr = np.array(scores)
        # subtract max for numerical stability
        scores_arr -= np.nanmax(scores_arr)
        exp_scores = np.exp(scores_arr)
        if np.isinf(exp_scores).any() or np.isnan(exp_scores).any():
            exp_scores = np.nan_to_num(exp_scores, nan=0.0, posinf=0.0, neginf=0.0)
        total = exp_scores.sum()
        if total == 0.0:
            return np.zeros(len(self.experts))
        return exp_scores / total

    def predict(
        self, history: Sequence[float], regime: float, caps: ResourceCapabilities
    ) -> float:
        """Combine expert predictions according to gating weights."""

        w = self.weights(regime, caps)
        preds = np.array([spec.model.predict(history) for spec in self.experts])
        return float(np.dot(w, preds))


__all__ = [
    "Expert",
    "TrendExpert",
    "MeanReversionExpert",
    "MacroExpert",
    "ExpertSpec",
    "GatingNetwork",
]
