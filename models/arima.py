from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from statsmodels.tsa.arima.model import ARIMA


@dataclass
class ARIMAModel:
    """Simple wrapper around :class:`statsmodels` ARIMA."""

    order: tuple[int, int, int] = (1, 0, 0)

    def fit(self, series: Sequence[float]) -> None:
        """Fit the ARIMA model to a 1D sequence."""
        arr = np.asarray(series, dtype=float)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("series must be a non-empty 1D array")
        self._model = ARIMA(arr, order=self.order).fit()

    def forecast(self, steps: int = 1) -> np.ndarray:
        """Forecast ``steps`` ahead returns."""
        if not hasattr(self, "_model"):
            raise RuntimeError("fit must be called before forecast")
        return self._model.forecast(steps=steps)


__all__ = ["ARIMAModel"]
