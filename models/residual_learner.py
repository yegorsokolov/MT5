"""Lightweight residual learner using LightGBM or a fallback regressor.

This module trains a model on ``[features, base_pred]`` to predict residuals
between a base model's predictions and the ground truth target.  The residual
model is stored alongside other model weights and can be used during inference
to refine predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
try:  # pragma: no cover - LightGBM is optional in tests
    from lightgbm import LGBMRegressor as _Regressor
except Exception:  # pragma: no cover - fallback to simple linear regressor
    class _Regressor:
        def fit(self, X: np.ndarray, y: np.ndarray) -> None:
            X_ext = np.hstack([X, np.ones((X.shape[0], 1))])
            beta, *_ = np.linalg.lstsq(X_ext, y, rcond=None)
            self.coef_ = beta[:-1]
            self.intercept_ = beta[-1]

        def predict(self, X: np.ndarray) -> np.ndarray:
            X_ext = np.hstack([X, np.ones((X.shape[0], 1))])
            return X_ext @ np.concatenate([self.coef_, [self.intercept_]])


_ROOT = Path(__file__).resolve().parents[1]
_REPORTS = _ROOT / "reports" / "residual_perf"
_DATA_CACHE = _ROOT / "reports" / "residual_data"


def _mse(y_true: Any, y_pred: Any) -> float:
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    return float(np.mean((y_t - y_p) ** 2))


def _model_path(model_name: str) -> Path:
    return Path(__file__).with_name(f"{model_name}_residual.pkl")


def _to_2d(arr: Any) -> np.ndarray:
    data = np.asarray(arr)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


def train(
    features: Any,
    base_pred: Any,
    target: Any,
    model_name: str,
) -> _Regressor:
    """Train a residual model and persist it to disk.

    ``features`` and ``base_pred`` are concatenated to form the training
    matrix.  The model learns to predict ``target - base_pred``.  Training
    metrics are written to ``reports/residual_perf``.
    """

    X = _to_2d(features)
    base = np.asarray(base_pred).reshape(-1, 1)
    y = np.asarray(target)
    X_full = np.hstack([X, base])

    model = _Regressor()
    model.fit(X_full, y - base.ravel())

    resid = model.predict(X_full)
    mse_base = _mse(y, base.ravel())
    mse_res = _mse(y, base.ravel() + resid)

    _REPORTS.mkdir(parents=True, exist_ok=True)
    with (_REPORTS / f"{model_name}.json").open("w") as f:
        json.dump({
            "mse_base": float(mse_base),
            "mse_with_residual": float(mse_res),
        }, f)

    joblib.dump(model, _model_path(model_name))
    return model


def load(model_name: str) -> _Regressor | None:
    """Load a residual model for ``model_name`` if available."""

    path = _model_path(model_name)
    if path.exists():
        return joblib.load(path)
    return None


def predict(features: Any, base_pred: Any, model: _Regressor) -> np.ndarray:
    """Return residual predictions for ``features`` and ``base_pred``."""

    X = _to_2d(features)
    base = np.asarray(base_pred).reshape(-1, 1)
    X_full = np.hstack([X, base])
    return model.predict(X_full)


def train_from_cache(model_name: str) -> None:
    """Train a residual model using cached data if available.

    Expects a joblib file at ``reports/residual_data/{model_name}.joblib``
    containing a dict with ``features``, ``base_pred`` and ``target`` arrays.
    """

    data_path = _DATA_CACHE / f"{model_name}.joblib"
    if not data_path.exists():  # pragma: no cover - no cached data in tests
        return
    data = joblib.load(data_path)
    train(data["features"], data["base_pred"], data["target"], model_name)


__all__ = ["train", "load", "predict", "train_from_cache"]
