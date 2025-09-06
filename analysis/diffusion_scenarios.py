"""Multi-asset diffusion based scenario generation.

This module trains a :class:`ScenarioDiffusion` model per asset and
aggregates the sampled paths to provide stress scenarios for strategy PnL.
It exposes helpers to draw crash, liquidity freeze and regime flip paths that
can be plugged into :mod:`stress_tests.scenario_runner`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .scenario_diffusion import ScenarioDiffusion


@dataclass
class MultiAssetDiffusion:
    """Train a diffusion model for each asset and aggregate samples.

    Parameters
    ----------
    seq_len:
        Window length used for each individual diffusion model.
    device:
        Device used for the underlying PyTorch models.
    """

    seq_len: int = 100
    device: str = "cpu"
    models: Dict[str, ScenarioDiffusion] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self, data: pd.DataFrame | Dict[str, Iterable[float]], epochs: int = 50
    ) -> None:
        """Fit a diffusion model for each asset column."""

        if isinstance(data, pd.DataFrame):
            items = data.items()
        else:
            items = data.items()  # type: ignore[assignment]
        for name, series in items:
            model = ScenarioDiffusion(seq_len=self.seq_len, device=self.device)
            model.fit(list(series), epochs=epochs)
            self.models[name] = model

    # ------------------------------------------------------------------
    # Sampling utilities
    # ------------------------------------------------------------------
    def _aggregate(
        self, paths: Dict[str, np.ndarray], weights: Optional[Sequence[float]]
    ) -> np.ndarray:
        if weights is None:
            weights = [1.0 / len(paths)] * len(paths)
        weights = np.asarray(list(weights), dtype=float)
        return np.sum([w * paths[k] for w, k in zip(weights, paths.keys())], axis=0)

    def generate(self, length: int, weights: Optional[Sequence[float]] = None) -> np.ndarray:
        """Generate an unconditional aggregated path."""

        paths = {k: m.generate(length) for k, m in self.models.items()}
        return self._aggregate(paths, weights)

    def sample_crash(
        self,
        length: int,
        crash: float = -0.3,
        recovery: float = 0.05,
        weights: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Sample a crash followed by recovery across all assets."""

        paths = {
            k: m.sample_crash_recovery(length, crash=crash, recovery=recovery)
            for k, m in self.models.items()
        }
        return self._aggregate(paths, weights)

    def sample_liquidity_freeze(
        self, length: int, freeze_days: int = 5, weights: Optional[Sequence[float]] = None
    ) -> np.ndarray:
        """Sample a path where initial observations are frozen to zero."""

        paths: Dict[str, np.ndarray] = {
            k: m.generate(length) for k, m in self.models.items()
        }
        for arr in paths.values():
            arr[:freeze_days] = 0.0
        return self._aggregate(paths, weights)

    def sample_regime_flip(
        self, length: int, weights: Optional[Sequence[float]] = None
    ) -> np.ndarray:
        """Sample a path where the second half flips sign (regime change)."""

        paths = {k: m.generate(length) for k, m in self.models.items()}
        half = length // 2
        for arr in paths.values():
            arr[half:] *= -1
        return self._aggregate(paths, weights)


__all__ = ["MultiAssetDiffusion"]

