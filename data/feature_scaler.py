from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class FeatureScaler:
    """Online feature scaler using Welford's algorithm.

    Maintains running mean and variance for each feature allowing
    normalisation of streaming data. The scaler can be updated with
    ``partial_fit`` and serialised to disk for later reuse.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.m2_: np.ndarray | None = None
        self.n_: int = 0
        self.columns_: list[str] | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame | np.ndarray, y: Any | None = None) -> "FeatureScaler":
        """Reset and fit on ``X``. Provided for sklearn compatibility."""
        self.mean_ = None
        self.m2_ = None
        self.n_ = 0
        self.columns_ = None
        return self.partial_fit(X)

    def partial_fit(self, X: pd.DataFrame | np.ndarray, y: Any | None = None) -> "FeatureScaler":
        """Update running statistics with a new batch ``X``."""
        if isinstance(X, pd.DataFrame):
            data = X.to_numpy(dtype=float)
            cols = list(X.columns)
        else:
            data = np.asarray(X, dtype=float)
            cols = list(range(data.shape[1]))

        if self.mean_ is None:
            self.mean_ = np.zeros(data.shape[1])
            self.m2_ = np.zeros(data.shape[1])
            self.columns_ = cols
        if self.columns_ != cols:
            raise ValueError("Feature columns do not match previously seen columns")

        batch_n = data.shape[0]
        if batch_n == 0:
            return self
        batch_mean = data.mean(axis=0)
        batch_m2 = ((data - batch_mean) ** 2).sum(axis=0)

        delta = batch_mean - self.mean_
        total_n = self.n_ + batch_n
        self.mean_ += delta * batch_n / total_n
        self.m2_ += batch_m2 + delta**2 * self.n_ * batch_n / total_n
        self.n_ = total_n
        return self

    # ------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Normalise ``X`` using the running statistics."""
        if self.mean_ is None or self.m2_ is None or self.n_ < 2:
            return X

        if isinstance(X, pd.DataFrame):
            data = X.to_numpy(dtype=float)
            cols = list(X.columns)
            if self.columns_ != cols:
                raise ValueError("Feature columns do not match previously seen columns")
            index = X.index
        else:
            data = np.asarray(X, dtype=float)
            cols = self.columns_
            index = None

        var = self.m2_ / max(self.n_ - 1, 1)
        std = np.sqrt(var)
        scaled = (data - self.mean_) / (std + 1e-8)

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(scaled, columns=cols, index=index)
        return scaled

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean_,
            "m2": self.m2_,
            "n": self.n_,
            "columns": self.columns_,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean_ = state.get("mean")
        self.m2_ = state.get("m2")
        self.n_ = state.get("n", 0)
        self.columns_ = state.get("columns")

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        joblib.dump(self.state_dict(), path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "FeatureScaler":
        scaler = cls()
        path = Path(path)
        if path.exists():
            state = joblib.load(path)
            scaler.load_state_dict(state)
        return scaler

