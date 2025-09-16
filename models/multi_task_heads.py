"""Lightweight multi-task estimator with shared feature trunk."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


class MultiTaskHeadEstimator(BaseEstimator, ClassifierMixin):
    """Neural-style estimator with a shared trunk and specialised heads."""

    def __init__(
        self,
        *,
        classification_targets: Iterable[str] | None = None,
        abs_targets: Iterable[str] | None = None,
        volatility_targets: Iterable[str] | None = None,
        hidden_dim: int = 32,
        learning_rate: float = 0.01,
        epochs: int = 200,
        classification_weight: float = 1.0,
        abs_weight: float = 1.0,
        volatility_weight: float = 1.0,
        l2: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        self.classification_targets = (
            list(classification_targets) if classification_targets else None
        )
        self.abs_targets = list(abs_targets) if abs_targets else None
        self.volatility_targets = list(volatility_targets) if volatility_targets else None
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.classification_weight = float(classification_weight)
        self.abs_weight = float(abs_weight)
        self.volatility_weight = float(volatility_weight)
        self.l2 = float(l2)
        self.random_state = random_state

        self._initialised = False
        self.thresholds_: Dict[str, float] = {}
        self.head_config_: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # sklearn compatibility helpers
    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "classification_targets": self.classification_targets,
            "abs_targets": self.abs_targets,
            "volatility_targets": self.volatility_targets,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "classification_weight": self.classification_weight,
            "abs_weight": self.abs_weight,
            "volatility_weight": self.volatility_weight,
            "l2": self.l2,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "MultiTaskHeadEstimator":
        for key, val in params.items():
            setattr(self, key, val)
        return self

    # ------------------------------------------------------------------
    # utility functions
    # ------------------------------------------------------------------
    def _rng(self) -> np.random.RandomState:
        seed = self.random_state if self.random_state is not None else 0
        return np.random.RandomState(seed)

    @staticmethod
    def _to_numpy(X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=float)
        return np.asarray(X, dtype=float)

    def _init_weights(self, input_dim: int) -> None:
        rng = self._rng()
        h = max(1, self.hidden_dim)
        limit = 1.0 / np.sqrt(max(input_dim, 1))
        self.W1_ = rng.uniform(-limit, limit, size=(input_dim, h))
        self.b1_ = np.zeros(h)

        def _head(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
            out_limit = 1.0 / np.sqrt(max(shape[0], 1))
            return (
                rng.uniform(-out_limit, out_limit, size=shape),
                np.zeros(shape[1]),
            )

        n_cls = len(self.classification_columns_)
        n_abs = len(self.abs_columns_)
        n_vol = len(self.vol_columns_)

        if n_cls:
            self.Wc_, self.bc_ = _head((h, n_cls))
        if n_abs:
            self.Wa_, self.ba_ = _head((h, n_abs))
        if n_vol:
            self.Wv_, self.bv_ = _head((h, n_vol))
        self._initialised = True

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    def _forward(
        self, X: pd.DataFrame | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        X_arr = self._to_numpy(X)
        H_raw = X_arr @ self.W1_ + self.b1_
        H = self._relu(H_raw)

        probs: np.ndarray | None = None
        abs_pred: np.ndarray | None = None
        vol_pred: np.ndarray | None = None
        if len(self.classification_columns_):
            logits = H @ self.Wc_ + self.bc_
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        if len(self.abs_columns_):
            abs_pred = H @ self.Wa_ + self.ba_
        if len(self.vol_columns_):
            vol_pred = H @ self.Wv_ + self.bv_
        return H, probs, abs_pred, vol_pred

    def _loss_and_grads(
        self,
        X: np.ndarray,
        H: np.ndarray,
        probs: np.ndarray | None,
        abs_pred: np.ndarray | None,
        vol_pred: np.ndarray | None,
        y_cls: np.ndarray | None,
        y_abs: np.ndarray | None,
        y_vol: np.ndarray | None,
        sample_weight: np.ndarray | None,
    ) -> tuple[float, Dict[str, np.ndarray]]:
        n = X.shape[0]
        if n == 0:
            return 0.0, {}
        sw = None if sample_weight is None else sample_weight.reshape(-1, 1)
        norm = float(sample_weight.sum()) if sample_weight is not None else float(n)

        total_loss = 0.0
        grads: Dict[str, np.ndarray] = {}
        dH = np.zeros_like(H)

        if probs is not None and y_cls is not None:
            probs_clip = np.clip(probs, 1e-6, 1 - 1e-6)
            if sw is None:
                ce = -(
                    y_cls * np.log(probs_clip)
                    + (1 - y_cls) * np.log(1 - probs_clip)
                )
                loss_cls = float(np.sum(ce) / norm)
                dlogits = (probs - y_cls) / norm
            else:
                ce = -sw * (
                    y_cls * np.log(probs_clip)
                    + (1 - y_cls) * np.log(1 - probs_clip)
                )
                loss_cls = float(np.sum(ce) / norm)
                dlogits = (probs - y_cls) * sw / norm
            total_loss += self.classification_weight * loss_cls
            dlogits *= self.classification_weight
            grads["Wc_"] = H.T @ dlogits + self.l2 * self.Wc_
            grads["bc_"] = dlogits.sum(axis=0)
            dH += dlogits @ self.Wc_.T

        if abs_pred is not None and y_abs is not None:
            diff_abs = abs_pred - y_abs
            if sw is None:
                loss_abs = 0.5 * float(np.sum(diff_abs**2) / norm)
                d_abs = diff_abs / norm
            else:
                loss_abs = 0.5 * float(np.sum(sw * diff_abs**2) / norm)
                d_abs = diff_abs * sw / norm
            total_loss += self.abs_weight * loss_abs
            d_abs *= self.abs_weight
            grads["Wa_"] = H.T @ d_abs + self.l2 * self.Wa_
            grads["ba_"] = d_abs.sum(axis=0)
            dH += d_abs @ self.Wa_.T

        if vol_pred is not None and y_vol is not None:
            diff_vol = vol_pred - y_vol
            if sw is None:
                loss_vol = 0.5 * float(np.sum(diff_vol**2) / norm)
                d_vol = diff_vol / norm
            else:
                loss_vol = 0.5 * float(np.sum(sw * diff_vol**2) / norm)
                d_vol = diff_vol * sw / norm
            total_loss += self.volatility_weight * loss_vol
            d_vol *= self.volatility_weight
            grads["Wv_"] = H.T @ d_vol + self.l2 * self.Wv_
            grads["bv_"] = d_vol.sum(axis=0)
            dH += d_vol @ self.Wv_.T

        mask = H > 0
        dZ = dH * mask
        grads["W1_"] = X.T @ dZ + self.l2 * self.W1_
        grads["b1_"] = dZ.sum(axis=0)
        return total_loss, grads

    # ------------------------------------------------------------------
    # core API
    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame | np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "MultiTaskHeadEstimator":
        if not isinstance(y, pd.DataFrame):
            raise TypeError("y must be a pandas DataFrame for multi-task training")

        y_df = y.copy()
        if self.classification_targets is None:
            self.classification_columns_ = [
                c for c in y_df.columns if c.startswith("direction_")
            ]
        else:
            self.classification_columns_ = list(self.classification_targets)
        if self.abs_targets is None:
            self.abs_columns_ = [c for c in y_df.columns if c.startswith("abs_return_")]
        else:
            self.abs_columns_ = list(self.abs_targets)
        if self.volatility_targets is None:
            self.vol_columns_ = [c for c in y_df.columns if c.startswith("volatility_")]
        else:
            self.vol_columns_ = list(self.volatility_targets)

        cols_needed: list[str] = []
        cols_needed.extend(self.classification_columns_)
        cols_needed.extend(self.abs_columns_)
        cols_needed.extend(self.vol_columns_)
        y_used = y_df[cols_needed] if cols_needed else y_df.iloc[:, :0]

        X_arr = self._to_numpy(X)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        self.input_dim_ = X_arr.shape[1]
        if not self._initialised:
            self._init_weights(self.input_dim_)

        y_cls = (
            y_used[self.classification_columns_].to_numpy(dtype=float)
            if self.classification_columns_
            else None
        )
        y_abs = (
            y_used[self.abs_columns_].to_numpy(dtype=float)
            if self.abs_columns_
            else None
        )
        y_vol = (
            y_used[self.vol_columns_].to_numpy(dtype=float)
            if self.vol_columns_
            else None
        )

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.shape[0] != X_arr.shape[0]:
                raise ValueError("sample_weight length mismatch")

        lr = self.learning_rate
        for _ in range(max(1, self.epochs)):
            H, probs, abs_pred, vol_pred = self._forward(X_arr)
            loss, grads = self._loss_and_grads(
                X_arr,
                H,
                probs,
                abs_pred,
                vol_pred,
                y_cls,
                y_abs,
                y_vol,
                sample_weight,
            )
            if not grads:
                break
            for name, grad in grads.items():
                np.clip(grad, -5.0, 5.0, out=grad)
                param = getattr(self, name)
                param -= lr * grad
                setattr(self, name, param)
            if np.isnan(loss) or np.isinf(loss):
                break

        self.primary_label_ = (
            self.classification_columns_[0] if self.classification_columns_ else None
        )
        self.head_config_ = {
            "classification": list(self.classification_columns_),
            "abs_return": list(self.abs_columns_),
            "volatility": list(self.vol_columns_),
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "l2": self.l2,
        }
        if self.classification_columns_:
            self.classes_ = np.array([0, 1], dtype=int)
        return self

    # ------------------------------------------------------------------
    # prediction helpers
    # ------------------------------------------------------------------
    def predict_classification_proba(
        self, X: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        if not self.classification_columns_:
            raise ValueError("No classification heads were trained")
        _, probs, _, _ = self._forward(X)
        if probs is None:
            raise ValueError("Classification probabilities are unavailable")
        return probs

    def predict_regression(
        self, X: pd.DataFrame | np.ndarray, head: str
    ) -> np.ndarray:
        _, _, abs_pred, vol_pred = self._forward(X)
        if head == "abs_return":
            if self.abs_columns_:
                if abs_pred is None:
                    raise ValueError("Abs return predictions unavailable")
                return abs_pred
            raise ValueError("No absolute return heads were trained")
        if head == "volatility":
            if self.vol_columns_:
                if vol_pred is None:
                    raise ValueError("Volatility predictions unavailable")
                return vol_pred
            raise ValueError("No volatility heads were trained")
        raise ValueError(f"Unknown head '{head}'")

    # sklearn API ------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> Any:
        if not self.classification_columns_:
            raise ValueError("No classification heads were trained")
        probs = self.predict_classification_proba(X)
        probs = _ensure_2d(probs)
        if probs.shape[1] == 1:
            p = probs[:, 0]
            return np.column_stack([1 - p, p])
        return [
            np.column_stack([1 - probs[:, i], probs[:, i]])
            for i in range(probs.shape[1])
        ]

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.classification_columns_:
            raise ValueError("No classification heads were trained")
        probs = self.predict_classification_proba(X)
        probs = _ensure_2d(probs)
        thresholds = [
            self.thresholds_.get(col, 0.5) for col in self.classification_columns_
        ]
        thr_arr = np.array(thresholds, dtype=float)
        return (probs >= thr_arr).astype(int)


__all__ = ["MultiTaskHeadEstimator"]

