"""Scaling utilities used by the synthetic data scripts."""
from __future__ import annotations

import numpy as np


class TimeSeriesMinMaxScaler:
    """Scale time-series features to a fixed range per feature.

    The implementation mirrors the small subset of the API used by the
    ``ydata-synthetic`` package so that the training scripts can remain mostly
    unchanged.  Inputs are expected to have shape ``(n_samples, seq_len,
    n_features)``.
    """

    def __init__(self, feature_range: tuple[float, float] = (-1.0, 1.0)) -> None:
        self.feature_range = feature_range
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.min_: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> "TimeSeriesMinMaxScaler":
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 3:
            raise ValueError(
                "Expected a 3D array with shape (n_samples, seq_len, n_features)."
            )

        self.data_min_ = data.min(axis=(0, 1))
        self.data_max_ = data.max(axis=(0, 1))
        data_range = self.data_max_ - self.data_min_
        data_range[data_range == 0] = 1.0

        feature_min, feature_max = self.feature_range
        if feature_min >= feature_max:
            raise ValueError("feature_range min must be less than max")

        self.scale_ = (feature_max - feature_min) / data_range
        self.min_ = feature_min - self.data_min_ * self.scale_
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.scale_ is None or self.min_ is None:
            raise RuntimeError("The scaler has not been fitted yet.")
        data = np.asarray(data, dtype=np.float32)
        scale = np.asarray(self.scale_, dtype=data.dtype)
        offset = np.asarray(self.min_, dtype=data.dtype)
        return data * scale + offset

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.scale_ is None or self.min_ is None or self.data_min_ is None:
            raise RuntimeError("The scaler has not been fitted yet.")
        data = np.asarray(data, dtype=np.float32)
        inv_scale = np.asarray(self.scale_, dtype=data.dtype)
        inv_scale = np.where(inv_scale == 0, 1.0, inv_scale)
        offset = np.asarray(self.min_, dtype=data.dtype)
        return (data - offset) / inv_scale

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)


class TimeSeriesScalerMinMax(TimeSeriesMinMaxScaler):
    """Backward compatible alias matching the ydata-synthetic API."""

    pass
