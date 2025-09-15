from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Tuple

import numpy as np

from telemetry import get_meter

meter = get_meter(__name__)
_interval_width = meter.create_histogram(
    "conformal_interval_width", description="Width of conformal prediction intervals"
)
_interval_coverage = meter.create_histogram(
    "conformal_interval_coverage", description="Interval coverage indicator"
)


@dataclass(slots=True)
class ConformalIntervalParams:
    """Container for calibrated conformal interval parameters."""

    alpha: float
    quantiles: float | Mapping[object, float]
    coverage: float | None = None
    coverage_by_regime: Mapping[object, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""

        def _normalise_key(key: object) -> object:
            if isinstance(key, (int, np.integer)):
                return int(key)
            try:
                return int(str(key))
            except (TypeError, ValueError):
                return str(key)

        data: dict[str, Any] = {"alpha": float(self.alpha)}
        q = self.quantiles
        if isinstance(q, Mapping):
            data["quantiles"] = {
                _normalise_key(k): float(v) for k, v in q.items()
            }
        else:
            data["quantiles"] = float(q)
        if self.coverage is not None:
            data["coverage"] = float(self.coverage)
        if self.coverage_by_regime:
            data["coverage_by_regime"] = {
                _normalise_key(k): float(v) for k, v in self.coverage_by_regime.items()
            }
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConformalIntervalParams":
        """Restore parameters from :meth:`to_dict` output."""

        def _restore_key(key: object) -> object:
            try:
                return int(key)  # handles numeric strings
            except (TypeError, ValueError):
                return key

        quantiles = data.get("quantiles", 0.0)
        if isinstance(quantiles, Mapping):
            qs: dict[object, float] = {
                _restore_key(k): float(v) for k, v in quantiles.items()
            }
        else:
            qs = float(quantiles)
        coverage_by_regime: Mapping[object, float] | None = None
        cbr = data.get("coverage_by_regime")
        if isinstance(cbr, Mapping):
            coverage_by_regime = {
                _restore_key(k): float(v) for k, v in cbr.items()
            }
        coverage = data.get("coverage")
        return cls(
            alpha=float(data.get("alpha", 0.1)),
            quantiles=qs,
            coverage=float(coverage) if coverage is not None else None,
            coverage_by_regime=coverage_by_regime,
        )

    def predict_interval(
        self,
        predictions: Iterable[float] | float,
        regimes: Iterable[object] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return interval bounds for ``predictions``."""

        return predict_interval(predictions, self.quantiles, regimes)


def fit_residuals(
    residuals: Iterable[float],
    alpha: float = 0.1,
    regimes: Iterable[object] | None = None,
) -> float | dict[object, float]:
    """Return (1-alpha) quantiles of absolute residuals."""

    arr = np.asarray(list(residuals), dtype=float)
    if arr.size == 0:
        return 0.0 if regimes is None else {}
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
    """Return lower and upper conformal prediction bounds."""

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
    """Return fraction of observations within intervals."""

    y = np.asarray(y_true, dtype=float)
    if y.size == 0:
        return 0.0
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    covered = (y >= lo) & (y <= hi)
    for c in covered.astype(float).ravel():
        _interval_coverage.record(float(c))
    return float(np.mean(covered))


def calibrate_intervals(
    y_true: Iterable[float],
    predictions: Iterable[float],
    alpha: float = 0.1,
    regimes: Iterable[object] | None = None,
) -> tuple[ConformalIntervalParams, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Calibrate conformal intervals from validation data."""

    preds = np.asarray(list(predictions), dtype=float)
    y_arr = np.asarray(list(y_true), dtype=float)
    residuals = y_arr - preds
    quantiles = fit_residuals(residuals, alpha=alpha, regimes=regimes)
    if preds.size == 0:
        params = ConformalIntervalParams(alpha=alpha, quantiles=quantiles)
        return params, residuals, (preds, preds)
    lower, upper = predict_interval(preds, quantiles, regimes)
    coverage = evaluate_coverage(y_arr, lower, upper)
    params = ConformalIntervalParams(alpha=alpha, quantiles=quantiles, coverage=coverage)
    return params, residuals, (lower, upper)
