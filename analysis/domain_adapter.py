"""Domain adaptation utilities.

This module provides :class:`DomainAdapter` which aligns feature
representations between a source training domain and a target live domain
using the CORrelation ALignment (CORAL) technique.  The adapter stores
means and covariance matrices for both domains and can transform incoming
features so that their distribution matches the historical training data.

The adapter also exposes a :meth:`reestimate` method that updates the
parameters for the target domain and logs alignment metrics via
``analytics.metrics_store``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

try:  # pragma: no cover - metrics_store optional in some tests
    from analytics.metrics_store import record_metric, TS_PATH
except Exception:  # pragma: no cover - fallback stub
    TS_PATH = Path("analytics/metrics_timeseries.parquet")

    def record_metric(*_: object, **__: object) -> None:  # type: ignore[override]
        return


@dataclass
class DomainAdapter:
    """Align feature distributions between source and target domains.

    The adapter stores the mean and covariance of the source (training)
    domain and optionally the current target (live) domain.  When
    :meth:`transform` is called, features are whitened using the target
    statistics and re-coloured with the source statistics as described in
    the CORAL method.
    """

    path: Optional[Path] = None
    source_mean_: Optional[np.ndarray] = None
    source_cov_: Optional[np.ndarray] = None
    target_mean_: Optional[np.ndarray] = None
    target_cov_: Optional[np.ndarray] = None
    columns_: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Fitting helpers
    # ------------------------------------------------------------------
    def fit_source(self, X: pd.DataFrame) -> None:
        """Store statistics for the source (training) domain."""
        numeric = X.select_dtypes(np.number)
        if numeric.empty:
            return
        self.columns_ = list(numeric.columns)
        self.source_mean_ = numeric.mean().to_numpy()
        self.source_cov_ = np.cov(numeric.to_numpy().T, bias=True)

    def update_target(self, X: pd.DataFrame) -> None:
        """Update statistics for the target (live) domain."""
        numeric = X.select_dtypes(np.number)
        if numeric.empty:
            return
        if self.columns_ and list(numeric.columns) != self.columns_:
            raise ValueError("Feature columns do not match adapter training columns")
        self.target_mean_ = numeric.mean().to_numpy()
        self.target_cov_ = np.cov(numeric.to_numpy().T, bias=True)

    # ------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align ``X`` to the source domain using CORAL.

        If either source or target statistics are unavailable the input is
        returned unchanged.
        """

        if (
            self.source_mean_ is None
            or self.source_cov_ is None
            or self.target_mean_ is None
            or self.target_cov_ is None
        ):
            return X

        data = X.to_numpy(dtype=float)
        # Compute matrix square roots via SVD for stability
        u_s, s_s, _ = np.linalg.svd(self.source_cov_)
        sqrt_source = u_s @ np.diag(np.sqrt(s_s + 1e-8)) @ u_s.T
        u_t, s_t, _ = np.linalg.svd(self.target_cov_)
        inv_sqrt_target = u_t @ np.diag(1.0 / np.sqrt(s_t + 1e-8)) @ u_t.T
        aligned = (data - self.target_mean_) @ inv_sqrt_target @ sqrt_source + self.source_mean_
        return pd.DataFrame(aligned, columns=self.columns_, index=X.index)

    # ------------------------------------------------------------------
    # Periodic re-estimation
    # ------------------------------------------------------------------
    def reestimate(self, X: pd.DataFrame, *, metrics_path: Path | None = None) -> None:
        """Recompute target statistics and log alignment metrics."""
        if metrics_path is None:
            metrics_path = TS_PATH
        if X.empty:
            return
        self.update_target(X)
        if (
            self.source_mean_ is not None
            and self.target_mean_ is not None
            and self.source_cov_ is not None
            and self.target_cov_ is not None
        ):
            mean_diff = float(np.linalg.norm(self.source_mean_ - self.target_mean_))
            cov_diff = float(np.linalg.norm(self.source_cov_ - self.target_cov_))
            record_metric("domain_adapter_mean_diff", mean_diff, path=metrics_path)
            record_metric("domain_adapter_cov_diff", cov_diff, path=metrics_path)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "source_mean": self.source_mean_,
            "source_cov": self.source_cov_,
            "target_mean": self.target_mean_,
            "target_cov": self.target_cov_,
            "columns": self.columns_,
        }

    def load_state_dict(self, state: dict) -> None:
        self.source_mean_ = state.get("source_mean")
        self.source_cov_ = state.get("source_cov")
        self.target_mean_ = state.get("target_mean")
        self.target_cov_ = state.get("target_cov")
        self.columns_ = state.get("columns")

    def save(self, path: Path | str | None = None) -> Path:
        path = Path(path or self.path or "domain_adapter.pkl")
        joblib.dump(self.state_dict(), path)
        self.path = path
        return path

    @classmethod
    def load(cls, path: Path | str | None = None) -> "DomainAdapter":
        path = Path(path or "domain_adapter.pkl")
        adapter = cls(path=path)
        if path.exists():
            state = joblib.load(path)
            adapter.load_state_dict(state)
        return adapter
