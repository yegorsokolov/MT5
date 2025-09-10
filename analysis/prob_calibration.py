"""Probability calibration utilities.

Provides Platt scaling and isotonic regression calibrators along with
helpers to plot reliability diagrams and log Brier scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from analysis.regime_thresholds import find_regime_thresholds


@dataclass
class ProbabilityCalibrator:
    """Fit Platt or isotonic calibrators for binary classification."""

    method: str = "platt"
    model: Any | None = None
    cv: int | None = None
    regime_thresholds: dict[int, float] | None = None

    def fit(
        self,
        y_true: np.ndarray,
        probs: np.ndarray | None = None,
        *,
        base_model: Any | None = None,
        X: Any | None = None,
        regimes: Iterable[int] | np.ndarray | None = None,
    ) -> "ProbabilityCalibrator":
        y_arr = np.asarray(y_true)
        if self.cv is not None:
            if base_model is None or X is None:
                raise ValueError("base_model and X required when cv is set")
            method = "sigmoid" if self.method == "platt" else "isotonic"
            ccv = CalibratedClassifierCV(base_model, cv=self.cv, method=method)
            ccv.fit(X, y_arr)
            self.model = ccv
            if regimes is not None:
                cal_probs = ccv.predict_proba(X)[:, 1]
                thr, _ = find_regime_thresholds(y_arr, cal_probs, regimes)
                setattr(ccv, "regime_thresholds", thr)
                self.regime_thresholds = thr

                def _predict(
                    X_new: Any,
                    *,
                    regimes: Iterable[int] | np.ndarray | None = None,
                    default_threshold: float = 0.5,
                ) -> np.ndarray:
                    probs = ccv.predict_proba(X_new)[:, 1]
                    if regimes is not None and thr:
                        preds = np.zeros(len(probs), dtype=int)
                        for reg, t in thr.items():
                            mask = np.asarray(regimes) == int(reg)
                            preds[mask] = (probs[mask] > t).astype(int)
                        return preds
                    return (probs > default_threshold).astype(int)

                setattr(ccv, "predict", _predict)
            return self

        if probs is None:
            raise ValueError("probs required when cv is None")
        p_arr = np.asarray(probs)
        if self.method == "platt":
            lr = LogisticRegression()
            lr.fit(p_arr.reshape(-1, 1), y_arr)
            self.model = lr
        elif self.method == "isotonic":
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(p_arr, y_arr)
            self.model = ir
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown calibration method: {self.method}")

        if regimes is not None:
            calibrated = self.predict(p_arr)
            thr, _ = find_regime_thresholds(y_arr, calibrated, regimes)
            setattr(self.model, "regime_thresholds", thr)
            self.regime_thresholds = thr
        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Calibrator has not been fitted")
        arr = np.asarray(data)
        if self.cv is not None:
            return self.model.predict_proba(arr)[:, 1]
        if self.method == "platt":
            return self.model.predict_proba(arr.reshape(-1, 1))[:, 1]
        return self.model.predict(arr)

    def predict_classes(
        self,
        data: Any,
        *,
        regimes: Iterable[int] | np.ndarray | None = None,
        default_threshold: float = 0.5,
    ) -> np.ndarray:
        """Return binary predictions applying regime-specific thresholds."""
        probs = self.predict(data)
        thr = self.regime_thresholds or getattr(self.model, "regime_thresholds", {})
        if regimes is not None and thr:
            preds = np.zeros(len(probs), dtype=int)
            r = np.asarray(regimes)
            for reg, t in thr.items():
                mask = r == int(reg)
                preds[mask] = (probs[mask] > t).astype(int)
            return preds
        return (probs > default_threshold).astype(int)


class CalibratedModel:
    """Wrap a model and apply probability calibration on predict_proba."""

    def __init__(self, model: Any, calibrator: ProbabilityCalibrator) -> None:
        self.model = model
        self.calibrator = calibrator
        self.regime_thresholds = calibrator.regime_thresholds or {}

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - delegation
        return getattr(self.model, item)

    def predict_proba(self, X: Any) -> np.ndarray:
        raw = self.model.predict_proba(X)
        if raw.ndim == 2 and raw.shape[1] == 2:
            calibrated = self.calibrator.predict(raw[:, 1])
            return np.column_stack([1 - calibrated, calibrated])
        return self.calibrator.predict(raw)

    def predict(
        self,
        X: Any,
        *,
        regimes: Iterable[int] | np.ndarray | None = None,
        default_threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict class labels using calibrated probabilities and thresholds."""
        probs = self.predict_proba(X)
        if probs.ndim == 2:
            probs = probs[:, 1]
        thr = self.regime_thresholds
        if regimes is not None and thr:
            preds = np.zeros(len(probs), dtype=int)
            r = np.asarray(regimes)
            for reg, t in thr.items():
                mask = r == int(reg)
                preds[mask] = (probs[mask] > t).astype(int)
            return preds
        return (probs > default_threshold).astype(int)


def plot_reliability_diagram(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    n_bins: int = 10,
    title: str | None = None,
    path: Path | None = None,
) -> None:
    """Create and optionally save a reliability diagram."""
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins)
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
    plt.close()


def log_reliability(
    y_true: np.ndarray,
    probs: np.ndarray,
    calibrated: np.ndarray,
    out_dir: Path,
    prefix: str,
    method: str,
) -> tuple[float, float]:
    """Log reliability diagrams and Brier scores."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_reliability_diagram(
        y_true,
        probs,
        title=f"{prefix} uncalibrated",
        path=out_dir / f"{prefix}_uncalibrated.png",
    )
    plot_reliability_diagram(
        y_true,
        calibrated,
        title=f"{prefix} calibrated ({method})",
        path=out_dir / f"{prefix}_{method}.png",
    )
    brier_raw = brier_score_loss(y_true, probs)
    brier_cal = brier_score_loss(y_true, calibrated)
    with (out_dir / f"{prefix}_brier.txt").open("w") as f:
        f.write(f"uncalibrated: {brier_raw}\n{method}: {brier_cal}\n")
    return brier_raw, brier_cal

