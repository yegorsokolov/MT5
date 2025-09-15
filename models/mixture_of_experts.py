"""Mixture-of-experts models with a gating network.

The module defines simple expert models and a gating network that assigns
probabilities to each expert based on the current market regime and the
available system resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Set

import numpy as np


@dataclass
class ResourceCapabilities:
    """Minimal resource capability specification.

    This mirrors the structure used in :mod:`utils.resource_monitor` but avoids
    importing heavy optional dependencies. Only the attributes required by the
    gating network are included.
    """

    cpus: int
    memory_gb: float
    has_gpu: bool
    gpu_count: int = 0
    gpu_model: str = ""
    cpu_flags: Set[str] | None = None


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

    def weights(
        self,
        regime: float,
        caps: ResourceCapabilities,
        diversity: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return softmax weights over experts.

        Parameters
        ----------
        regime:
            Market regime label used by the gating network.
        caps:
            Available :class:`~utils.resource_monitor.ResourceCapabilities`.
        diversity:
            Optional pre-computed weights favouring experts with low error
            correlation. If provided, the softmax weights are multiplied by these
            values and renormalised. This allows upstream training code to
            down-weight highly correlated experts.
        """

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
            base = np.ones(len(self.experts)) / len(self.experts)
        else:
            base = exp_scores / total
        if diversity is not None:
            div = np.asarray(diversity, dtype=float)
            if div.shape != base.shape:
                raise ValueError("diversity weight size mismatch")
            base = base * div
            total = base.sum()
            if total <= 0 or not np.isfinite(total):
                base = np.ones(len(self.experts)) / len(self.experts)
            else:
                base = base / total
        return base

    def predict(
        self,
        history: Sequence[float],
        regime: float,
        caps: ResourceCapabilities,
        diversity: np.ndarray | None = None,
    ) -> float:
        """Combine expert predictions according to gating weights."""

        w = self.weights(regime, caps, diversity)
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
