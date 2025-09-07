"""Residual stacker to model errors on top of a base predictor."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:  # pragma: no cover - optional dependency
    from lightgbm import LGBMRegressor as _Regressor
except Exception:  # pragma: no cover - fallback linear model
    class _Regressor:
        """Simple linear regressor used when LightGBM is unavailable."""

        def fit(self, X: np.ndarray, y: np.ndarray) -> None:
            X_ext = np.hstack([X, np.ones((X.shape[0], 1))])
            beta, *_ = np.linalg.lstsq(X_ext, y, rcond=None)
            self.coef_ = beta[:-1]
            self.intercept_ = beta[-1]

        def predict(self, X: np.ndarray) -> np.ndarray:
            X_ext = np.hstack([X, np.ones((X.shape[0], 1))])
            return X_ext @ np.concatenate([self.coef_, [self.intercept_]])


def _model_path(name: str) -> Path:
    """Return path for persisted residual stacker model."""

    return Path(__file__).with_name(f"{name}_stacker.pkl")


def _to_2d(arr: Any) -> np.ndarray:
    data = np.asarray(arr)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


def train(features: Any, base_pred: Any, target: Any, model_name: str) -> _Regressor:
    """Fit a secondary model on ``features`` and ``base_pred`` to learn residuals."""

    X = _to_2d(features)
    base = np.asarray(base_pred).reshape(-1, 1)
    y = np.asarray(target)
    X_full = np.hstack([X, base])

    model = _Regressor()
    model.fit(X_full, y - base.ravel())
    joblib.dump(model, _model_path(model_name))
    return model


def load(model_name: str) -> _Regressor | None:
    """Load a previously trained residual stacker for ``model_name`` if present."""

    path = _model_path(model_name)
    if path.exists():
        return joblib.load(path)
    return None


def predict(features: Any, base_pred: Any, model: _Regressor) -> np.ndarray:
    """Predict residual errors for ``features`` and ``base_pred`` using ``model``."""

    X = _to_2d(features)
    base = np.asarray(base_pred).reshape(-1, 1)
    X_full = np.hstack([X, base])
    return model.predict(X_full)


__all__ = ["train", "load", "predict"]
