"""Model ensembling utilities.

This module provides an :class:`EnsembleModel` that can combine predictions
from heterogeneous models such as gradient boosting, transformers and
reinforcement learning policies. Predictions are combined either via a simple
weighted average or by using a provided meta model for stacking.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import logging
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

try:  # pragma: no cover - fallback when package import fails
    from analysis.ensemble_diversity import error_correlation_matrix
except Exception:  # noqa: BLE001
    spec = importlib.util.spec_from_file_location(
        "ensemble_diversity",
        Path(__file__).resolve().parents[1] / "analysis" / "ensemble_diversity.py",
    )
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)  # type: ignore[assignment]
    error_correlation_matrix = _mod.error_correlation_matrix

logger = logging.getLogger(__name__)


@dataclass
class EnsembleModel:
    """Combine predictions from multiple models.

    Parameters
    ----------
    models:
        Mapping of model name to the model object or a callable that accepts a
        :class:`pandas.DataFrame` and returns a 1D array of probabilities.
    weights:
        Optional mapping of model name to ensemble weight. If not provided a
        uniform weighting is used.
    meta_model:
        Optional meta estimator implementing ``predict_proba``. When supplied
        stacking is used instead of weighted averaging.
    """

    models: Mapping[str, Any]
    weights: Optional[Mapping[str, float]] = None
    meta_model: Any | None = None

    def _predict_single(self, model: Any, df: pd.DataFrame) -> np.ndarray:
        """Return probability predictions for a single model."""
        # Accept callables returning probabilities directly
        if callable(model):
            return np.asarray(model(df))
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)
            probs = np.asarray(probs)
            if probs.ndim == 2:
                # assume positive class is the last column
                probs = probs[:, -1]
            return probs
        if hasattr(model, "predict"):
            return np.asarray(model.predict(df))
        if hasattr(model, "predict_proba_one"):
            # river style models
            return np.array([
                model.predict_proba_one(row).get(1, 0.0)
                for row in df.to_dict("records")
            ])
        raise AttributeError("Model does not support prediction: %r" % (model,))

    def predict(
        self, df: pd.DataFrame, y_true: Iterable[float] | None = None
    ) -> Dict[str, np.ndarray]:
        """Return per-model and ensemble probabilities for ``df``.

        If ``y_true`` is provided, error correlations are used to favour less
        correlated models. Resulting weights and validation metric improvements
        are logged.
        """

        per_model = {
            name: self._predict_single(model, df)
            for name, model in self.models.items()
        }
        arr = np.vstack(list(per_model.values()))
        names = list(per_model.keys())
        truth = np.asarray(y_true) if y_true is not None else None

        if self.meta_model is not None:
            meta_X = arr.T
            ensemble = self.meta_model.predict_proba(meta_X)
            ensemble = np.asarray(ensemble)
            if ensemble.ndim == 2:
                ensemble = ensemble[:, -1]
        else:
            if self.weights:
                base_w = np.array([self.weights.get(name, 1.0) for name in names])
                base_w = base_w / base_w.sum()
            else:
                base_w = np.ones(len(per_model)) / len(per_model)
            w = base_w

            if truth is not None and len(per_model) > 1:
                corr = error_correlation_matrix(per_model, truth).abs()
                np.fill_diagonal(corr.values, np.nan)
                mean_corr = np.nanmean(corr.values, axis=1)
                diversity = np.clip(1 - mean_corr, 0.0, None)
                w = base_w * diversity
                if np.allclose(w.sum(), 0.0):
                    w = base_w
                else:
                    w = w / w.sum()
                logger.info(
                    "Diversity-adjusted weights: %s",
                    {n: float(val) for n, val in zip(names, w)},
                )

            ensemble = np.average(arr, axis=0, weights=w)

            if truth is not None and len(per_model) > 1:
                baseline = np.average(arr, axis=0, weights=base_w)
                base_mse = float(np.mean((baseline - truth) ** 2))
                adj_mse = float(np.mean((ensemble - truth) ** 2))
                logger.info(
                    "Validation MSE: %.6f -> %.6f (Δ=%.6f)",
                    base_mse,
                    adj_mse,
                    base_mse - adj_mse,
                )

        per_model["ensemble"] = ensemble
        return per_model
