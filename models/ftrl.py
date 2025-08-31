"""Lightweight Follow-The-Regularized-Leader linear model.

This module implements a simple online logistic regression using the
Follow-The-Regularized-Leader (FTRL) algorithm.  The implementation is
self-contained and designed for environments where computational resources are
limited.  The model supports incremental updates and can be used as a drop-in
replacement for heavier models when running on "lite" capability tiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Iterable

import numpy as np


@dataclass
class FTRLModel:
    """Linear classifier trained with the FTRL-Proximal algorithm.

    Parameters follow the notation from the "Ad Click Prediction: a View from
    the Trenches" paper.  ``alpha`` controls the learning rate while ``beta``
    provides smoothing for the adaptive rates.  ``l1`` and ``l2`` correspond to
    L1 and L2 regularisation strengths respectively.
    """

    alpha: float = 0.1
    beta: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    dim: int = 0
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    def __post_init__(self) -> None:
        self.z = np.zeros(self.dim, dtype=float)
        self.n = np.zeros(self.dim, dtype=float)
        self.w = np.zeros(self.dim, dtype=float)

    # ------------------------------------------------------------------
    def _ensure_dim(self, dim: int) -> None:
        if dim <= self.dim:
            return
        extra = dim - self.dim
        self.z = np.concatenate([self.z, np.zeros(extra)])
        self.n = np.concatenate([self.n, np.zeros(extra)])
        self.w = np.concatenate([self.w, np.zeros(extra)])
        self.dim = dim

    def _update_weights(self) -> None:
        for i in range(self.dim):
            z = self.z[i]
            if abs(z) <= self.l1:
                self.w[i] = 0.0
            else:
                sign = -1.0 if z < 0 else 1.0
                self.w[i] = (
                    sign * self.l1 - z
                ) / ((self.beta + math.sqrt(self.n[i])) / self.alpha + self.l2)

    # ------------------------------------------------------------------
    def predict_proba(self, x: Iterable[float]) -> float:
        x_arr = np.asarray(x, dtype=float)
        self._ensure_dim(len(x_arr))
        self._update_weights()
        wx = float(np.dot(self.w, x_arr))
        # Numerically stable sigmoid
        if wx >= 0:
            z = math.exp(-wx)
            return 1.0 / (1.0 + z)
        z = math.exp(wx)
        return z / (1.0 + z)

    def predict(self, x: Iterable[float]) -> float:
        return self.predict_proba(x)

    # ------------------------------------------------------------------
    def update(self, x: Iterable[float], y: float) -> None:
        """Update model weights with a single training example.

        ``x`` should be an iterable of feature values and ``y`` the binary label
        (``0`` or ``1``).  Convergence metrics such as logistic loss and weight
        norm are logged via ``analytics.metrics_store.record_metric`` when
        available.
        """

        x_arr = np.asarray(x, dtype=float)
        self._ensure_dim(len(x_arr))
        p = self.predict_proba(x_arr)
        g = (p - y) * x_arr
        sigma = (np.sqrt(self.n + g * g) - np.sqrt(self.n)) / self.alpha
        self.z += g - sigma * self.w
        self.n += g * g

        loss = -(
            y * math.log(p + 1e-15) + (1.0 - y) * math.log(1.0 - p + 1e-15)
        )
        try:  # pragma: no cover - metrics module optional
            from analytics.metrics_store import record_metric

            record_metric("ftrl_loss", float(loss))
            record_metric("ftrl_weight_norm", float(np.linalg.norm(self.w)))
        except Exception:
            pass
        self.logger.debug("FTRL update: loss=%.6f", loss)
