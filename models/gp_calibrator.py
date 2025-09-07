from __future__ import annotations

import numpy as np
from dataclasses import dataclass


def _to_2d(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


def _rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float) -> np.ndarray:
    """Return RBF kernel matrix between ``x1`` and ``x2``."""
    x1 = _to_2d(x1)
    x2 = _to_2d(x2)
    sq1 = np.sum(x1 ** 2, axis=1)[:, None]
    sq2 = np.sum(x2 ** 2, axis=1)
    sqdist = sq1 + sq2 - 2 * x1 @ x2.T
    return np.exp(-0.5 / (length_scale ** 2) * sqdist)


@dataclass
class GPCalibrator:
    """Calibrate predictive variance using a simple Gaussian Process.

    The implementation here is a lightweight 1D Gaussian Process that maps model
    predictions to squared residuals.  It is designed to avoid heavy external
    dependencies while providing uncertainty estimates for risk management.
    """

    length_scale: float = 1.0
    noise: float = 1e-4
    X_train: np.ndarray | None = None
    y_train: np.ndarray | None = None
    K_inv: np.ndarray | None = None
    scale: float = 1.0
    z_score: float = 1.96

    def fit(self, preds: np.ndarray, targets: np.ndarray) -> "GPCalibrator":
        """Fit the GP on validation residuals."""
        X = _to_2d(preds)
        y = (np.asarray(targets, dtype=float) - np.asarray(preds, dtype=float)) ** 2
        if self.length_scale <= 0:
            diffs = np.abs(np.subtract.outer(X.ravel(), X.ravel()))
            self.length_scale = float(np.median(diffs)) or 1.0
        K = _rbf_kernel(X, X, self.length_scale)
        K += np.eye(len(X)) * self.noise
        self.K_inv = np.linalg.inv(K)
        self.X_train = X
        self.y_train = y

        K_s = _rbf_kernel(X, X, self.length_scale)
        mu = K_s @ self.K_inv @ y
        k_ss = _rbf_kernel(X, X, self.length_scale) + self.noise
        cov = k_ss - K_s @ self.K_inv @ K_s.T
        var_train = mu + np.clip(np.diag(cov), 0.0, None)
        mean_var = float(np.mean(var_train))
        mean_y = float(np.mean(y))
        if mean_var > 0:
            self.scale = mean_y / mean_var
        std_train = np.sqrt(var_train * self.scale)
        ratio = np.sqrt(y) / std_train
        self.z_score = float(np.quantile(ratio, 0.95))
        return self

    def predict_variance(self, preds: np.ndarray) -> np.ndarray:
        """Return calibrated predictive variance for ``preds``."""
        if self.K_inv is None or self.X_train is None or self.y_train is None:
            raise RuntimeError("GPCalibrator must be fitted before prediction")
        X = _to_2d(preds)
        K_s = _rbf_kernel(X, self.X_train, self.length_scale)
        mu = K_s @ self.K_inv @ self.y_train
        k_ss = _rbf_kernel(X, X, self.length_scale) + self.noise
        cov = k_ss - K_s @ self.K_inv @ K_s.T
        var = mu + np.clip(np.diag(cov), 0.0, None)
        var = var * self.scale
        return np.maximum(var, 1e-9)

    def predict_interval(self, preds: np.ndarray, z: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return lower and upper ``z``-score credible interval bounds."""
        preds_arr = np.asarray(preds, dtype=float)
        var = self.predict_variance(preds_arr)
        std = np.sqrt(var)
        z_val = z if z is not None else self.z_score
        lower = preds_arr - z_val * std
        upper = preds_arr + z_val * std
        return lower, upper


__all__ = ["GPCalibrator"]
