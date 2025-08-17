from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple

from telemetry import get_meter

meter = get_meter(__name__)
_interval_width = meter.create_histogram(
    "conformal_interval_width", description="Width of conformal prediction intervals"
)
_interval_coverage = meter.create_histogram(
    "conformal_interval_coverage", description="Interval coverage indicator"
)

def fit_residuals(residuals: Iterable[float], alpha: float = 0.1) -> float:
    """Return the (1-alpha) quantile of absolute residuals."""
    arr = np.asarray(list(residuals), dtype=float)
    return float(np.quantile(np.abs(arr), 1 - alpha))

def predict_interval(predictions: Iterable[float] | float, q: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return lower and upper conformal prediction bounds.

    Parameters
    ----------
    predictions: Iterable[float] | float
        Model predictions.
    q: float
        Quantile of residuals from :func:`fit_residuals`.
    """
    preds = np.asarray(predictions, dtype=float)
    lower = preds - q
    upper = preds + q
    widths = upper - lower
    for w in np.atleast_1d(widths):
        _interval_width.record(float(w))
    return lower, upper

def evaluate_coverage(
    y_true: Iterable[float] | float,
    lower: Iterable[float] | float,
    upper: Iterable[float] | float,
) -> float:
    """Return fraction of observations within intervals.

    Also logs coverage to Prometheus."""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    covered = (y >= lo) & (y <= hi)
    for c in covered.astype(float).ravel():
        _interval_coverage.record(float(c))
    return float(np.mean(covered))
