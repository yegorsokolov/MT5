"""Mixture-of-experts models with a gating network."""

from __future__ import annotations

from collections.abc import Mapping
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

    def _resolve_diversity(
        self,
        diversity: np.ndarray
        | Mapping[float | int | str, Sequence[float]]
        | None,
        regime: float,
    ) -> np.ndarray | None:
        if diversity is None:
            return None
        if isinstance(diversity, Mapping):
            candidates: list[float | int | str] = [regime]
            try:
                candidates.append(int(regime))
            except Exception:
                pass
            try:
                candidates.append(float(regime))
            except Exception:
                pass
            candidates.append("default")
            vec = None
            for key in candidates:
                if key in diversity:
                    vec = diversity[key]
                    break
            if vec is None and diversity:
                vec = next(iter(diversity.values()))
        else:
            vec = diversity
        if vec is None:
            return None
        arr = np.asarray(vec, dtype=float).reshape(-1)
        if arr.shape[0] != len(self.experts):
            raise ValueError("diversity weight size mismatch")
        return arr

    @staticmethod
    def _normalise(vec: Sequence[float]) -> np.ndarray:
        arr = np.asarray(vec, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0.0, None)
        total = arr.sum()
        if total <= 0.0 or not np.isfinite(total):
            return np.ones_like(arr) / len(arr)
        return arr / total

    @staticmethod
    def _apply_budgets(weights: np.ndarray, budgets: Sequence[float]) -> np.ndarray:
        arr = np.asarray(budgets, dtype=float)
        if arr.shape != weights.shape:
            raise ValueError("risk budget size mismatch")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0.0, None)
        total_budget = arr.sum()
        if total_budget <= 0.0 or not np.isfinite(total_budget):
            return np.ones_like(weights) / len(weights)
        arr = arr / total_budget
        pref = GatingNetwork._normalise(weights)
        allocation = np.zeros_like(pref)
        capacity = arr.copy()
        leftover = 1.0
        active = np.arange(len(pref))
        while leftover > 1e-12 and active.size > 0:
            pref_slice = pref[active]
            slice_total = pref_slice.sum()
            if slice_total <= 0.0 or not np.isfinite(slice_total):
                proportion = np.ones_like(pref_slice) / pref_slice.size
            else:
                proportion = pref_slice / slice_total
            step = np.minimum(capacity[active], leftover * proportion)
            allocation[active] += step
            leftover -= float(step.sum())
            capacity[active] -= step
            active = active[capacity[active] > 1e-12]
        total = allocation.sum()
        if total <= 0.0:
            allocation = arr / arr.sum()
        else:
            allocation /= total
        return allocation

    def weights(
        self,
        regime: float,
        caps: ResourceCapabilities,
        diversity: np.ndarray
        | Mapping[float | int | str, Sequence[float]]
        | None = None,
        risk_budgets: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Return softmax weights over experts respecting optional budgets."""

        scores = []
        for idx, spec in enumerate(self.experts):
            if not self._supported(caps, spec.requirements):
                scores.append(-np.inf)
            else:
                scores.append(-self.sharpness * abs(regime - idx))
        scores_arr = np.array(scores)
        scores_arr -= np.nanmax(scores_arr)
        exp_scores = np.exp(scores_arr)
        if np.isinf(exp_scores).any() or np.isnan(exp_scores).any():
            exp_scores = np.nan_to_num(exp_scores, nan=0.0, posinf=0.0, neginf=0.0)
        base = self._normalise(exp_scores)
        div = self._resolve_diversity(diversity, regime)
        if div is not None:
            base = self._normalise(base * np.clip(div, 0.0, None))
        if risk_budgets is not None:
            base = self._apply_budgets(base, risk_budgets)
        return base

    def predict(
        self,
        history: Sequence[float],
        regime: float,
        caps: ResourceCapabilities,
        diversity: np.ndarray
        | Mapping[float | int | str, Sequence[float]]
        | None = None,
        risk_budgets: Sequence[float] | None = None,
    ) -> float:
        """Combine expert predictions according to gating weights."""

        w = self.weights(regime, caps, diversity, risk_budgets=risk_budgets)
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
