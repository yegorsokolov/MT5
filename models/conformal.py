from __future__ import annotations

import numpy as np
from typing import Iterable, Mapping, Tuple

from telemetry import get_meter

meter = get_meter(__name__)
_interval_width = meter.create_histogram(
    "conformal_interval_width", description="Width of conformal prediction intervals"
)
_interval_coverage = meter.create_histogram(
    "conformal_interval_coverage", description="Interval coverage indicator"
)

def fit_residuals(
    residuals: Iterable[float],
    alpha: float = 0.1,
    regimes: Iterable[object] | None = None,
) -> float | dict[object, float]:
    """Return (1-alpha) quantiles of absolute residuals.

    When ``regimes`` is provided, residuals are segmented by regime and a
    dictionary mapping each regime to its quantile is returned.  Otherwise a
    single quantile for all residuals is produced.
    """
    arr = np.asarray(list(residuals), dtype=float)
    abs_arr = np.abs(arr)
    if regimes is None:
        return float(np.quantile(abs_arr, 1 - alpha))
    reg_arr = np.asarray(list(regimes))
    if len(reg_arr) != len(abs_arr):  # pragma: no cover - sanity check
        raise ValueError("residuals and regimes must have the same length")
    qs: dict[object, float] = {}
    for reg in np.unique(reg_arr):
        qs[reg] = float(np.quantile(abs_arr[reg_arr == reg], 1 - alpha))
    return qs

def predict_interval(
    predictions: Iterable[float] | float,
    q: float | Mapping[object, float],
    regimes: Iterable[object] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return lower and upper conformal prediction bounds.

    Parameters
    ----------
    predictions: Iterable[float] | float
        Model predictions.
    q: float | Mapping[object, float]
        Quantile(s) of residuals from :func:`fit_residuals`. When a mapping is
        provided, ``regimes`` must also be supplied to match each prediction to
        its regime-specific quantile.
    regimes: Iterable[object] | None
        Sequence of regime identifiers corresponding to ``predictions`` when
        ``q`` is a mapping.
    """
    preds = np.asarray(predictions, dtype=float)
    if isinstance(q, Mapping):
        if regimes is None:
            raise ValueError("regimes must be provided when q is a mapping")
        reg_arr = np.asarray(list(regimes))
        q_arr = np.array([q[reg] for reg in reg_arr], dtype=float)
    else:
        q_arr = q
    lower = preds - q_arr
    upper = preds + q_arr
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
