"""Lightweight preprocessing utilities shared by training and inference."""

from __future__ import annotations

from typing import Any, Dict, Iterable

try:  # pragma: no cover - optional dependency in lightweight test envs
    import numpy as np
except Exception:  # pragma: no cover - provide minimal fallbacks
    np = None  # type: ignore[assignment]

import pandas as pd
from pandas.api import types as pd_types

__all__ = ["FeatureSanitizer"]


if np is not None:  # pragma: no cover - exercised in main pipeline
    _INF_POS = np.inf
    _INF_NEG = -np.inf

    def _nanmedian(values: Iterable[float]) -> float:
        return float(np.nanmedian(values))

    def _nanmean(values: Iterable[float]) -> float:
        return float(np.nanmean(values))

else:  # pragma: no cover - executed when numpy is unavailable in tests
    _INF_POS = float("inf")
    _INF_NEG = float("-inf")

    def _nanmedian(values: Iterable[float]) -> float:
        data = [v for v in values if v == v]
        if not data:
            return float("nan")
        data.sort()
        mid = len(data) // 2
        if len(data) % 2:
            return float(data[mid])
        return float((data[mid - 1] + data[mid]) / 2)

    def _nanmean(values: Iterable[float]) -> float:
        data = [v for v in values if v == v]
        if not data:
            return float("nan")
        return float(sum(data) / len(data))


class FeatureSanitizer:
    """Coerce model features to numeric values and fill missing data."""

    def __init__(
        self,
        *,
        fill_method: str = "median",
        fill_value: float | None = None,
    ) -> None:
        self.fill_method = str(fill_method).lower()
        self.fill_value = fill_value
        self.expected_columns_: list[str] = []
        self.fill_values_: dict[str, float] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # sklearn-style helpers
    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {"fill_method": self.fill_method, "fill_value": self.fill_value}

    def set_params(self, **params: Any) -> "FeatureSanitizer":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_frame(X: pd.DataFrame | np.ndarray | Dict[str, Any]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, dict):
            return pd.DataFrame(X)
        return pd.DataFrame(X)

    @staticmethod
    def _coerce_numeric(series: pd.Series) -> pd.Series:
        if not pd_types.is_numeric_dtype(series):
            series = pd.to_numeric(series, errors="coerce")
        nan_value = np.nan if np is not None else float("nan")
        series = series.replace([_INF_POS, _INF_NEG], nan_value)
        return series.astype(float)

    def _compute_fill_value(self, series: pd.Series) -> float:
        cleaned = series.dropna()
        if cleaned.empty:
            base = self.fill_value if self.fill_value is not None else 0.0
            return float(base)
        values = cleaned.to_numpy(dtype=float) if np is not None else cleaned.astype(float).tolist()
        if self.fill_method == "median":
            fill = _nanmedian(values)
        elif self.fill_method == "mean":
            fill = _nanmean(values)
        elif self.fill_method == "zero":
            fill = 0.0
        elif self.fill_method == "constant":
            base = self.fill_value if self.fill_value is not None else 0.0
            fill = float(base)
        else:
            raise ValueError(f"Unknown fill_method '{self.fill_method}'")
        if np is not None:
            if not np.isfinite(fill):
                fill = 0.0
        else:
            if fill != fill or fill in (_INF_POS, _INF_NEG):
                fill = 0.0
        return fill

    # ------------------------------------------------------------------
    # core API
    # ------------------------------------------------------------------
    def fit(
        self, X: pd.DataFrame | np.ndarray | Dict[str, Any], y: Any = None
    ) -> "FeatureSanitizer":
        frame = self._ensure_frame(X)
        self.expected_columns_ = list(frame.columns)
        self.fill_values_ = {}
        for col in self.expected_columns_:
            series = self._coerce_numeric(frame[col])
            self.fill_values_[col] = self._compute_fill_value(series)
        self._fitted = True
        return self

    def transform(
        self, X: pd.DataFrame | np.ndarray | Dict[str, Any]
    ) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FeatureSanitizer must be fitted before calling transform")
        frame = self._ensure_frame(X)
        nan_value = np.nan if np is not None else float("nan")
        for col in self.expected_columns_:
            if col not in frame.columns:
                frame[col] = nan_value
        frame = frame.loc[:, self.expected_columns_]
        for col in self.expected_columns_:
            series = self._coerce_numeric(frame[col])
            fill = self.fill_values_.get(col, 0.0)
            frame[col] = series.fillna(fill)
        return frame.astype(float)

    def fit_transform(
        self, X: pd.DataFrame | np.ndarray | Dict[str, Any], y: Any = None
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------
    # persistence helpers
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        if not self._fitted:
            return {
                "fill_method": self.fill_method,
                "fill_value": self.fill_value,
                "expected_columns": [],
                "fill_values": {},
            }
        return {
            "fill_method": self.fill_method,
            "fill_value": self.fill_value,
            "expected_columns": list(self.expected_columns_),
            "fill_values": {k: float(v) for k, v in self.fill_values_.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "FeatureSanitizer":
        self.fill_method = str(state.get("fill_method", self.fill_method)).lower()
        self.fill_value = state.get("fill_value", self.fill_value)
        self.expected_columns_ = list(state.get("expected_columns", []))
        fill_values = state.get("fill_values", {})
        self.fill_values_ = {k: float(v) for k, v in fill_values.items()}
        self._fitted = bool(self.expected_columns_)
        return self

