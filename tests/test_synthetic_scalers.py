"""Compatibility tests for the synthetic scaling utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

np = pytest.importorskip("numpy")

if TYPE_CHECKING:
    import numpy as np

from synthetic.scalers import TimeSeriesMinMaxScaler, TimeSeriesScalerMinMax


def _sample_data() -> "np.ndarray":
    base = np.array(
        [
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
            [[-1.0, 0.5], [0.5, 1.5], [1.5, 2.5]],
        ],
        dtype=np.float32,
    )
    return base


def test_minmax_scaler_round_trip() -> None:
    data = _sample_data()
    scaler = TimeSeriesMinMaxScaler()
    scaled = scaler.fit_transform(data)
    restored = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(restored, data, rtol=1e-6, atol=1e-6)


def test_legacy_alias_matches_primary_scaler() -> None:
    data = _sample_data()
    legacy = TimeSeriesScalerMinMax()
    modern = TimeSeriesMinMaxScaler()

    legacy_scaled = legacy.fit_transform(data)
    modern_scaled = modern.fit_transform(data)

    np.testing.assert_allclose(legacy_scaled, modern_scaled, rtol=1e-6, atol=1e-6)
