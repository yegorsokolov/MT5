"""Model ensembling utilities.

This module provides an :class:`EnsembleModel` that can combine predictions
from heterogeneous models such as gradient boosting, transformers and
reinforcement learning policies. Predictions are combined either via a simple
weighted average or by using a provided meta model for stacking.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


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

    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Return per-model and ensemble probabilities for ``df``."""
        per_model = {
            name: self._predict_single(model, df)
            for name, model in self.models.items()
        }
        arr = np.vstack(list(per_model.values()))

        if self.meta_model is not None:
            meta_X = arr.T
            ensemble = self.meta_model.predict_proba(meta_X)
            ensemble = np.asarray(ensemble)
            if ensemble.ndim == 2:
                ensemble = ensemble[:, -1]
        else:
            if self.weights:
                w = np.array([self.weights.get(name, 1.0) for name in per_model])
                w = w / w.sum()
            else:
                w = np.ones(len(per_model)) / len(per_model)
            ensemble = np.average(arr, axis=0, weights=w)

        per_model["ensemble"] = ensemble
        return per_model
