"""Probability calibration utilities.

Provides Platt scaling and isotonic regression calibrators along with
helpers to plot reliability diagrams and log Brier scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


@dataclass
class ProbabilityCalibrator:
    """Fit Platt or isotonic calibrators for binary classification."""

    method: str = "platt"
    model: Any | None = None

    def fit(self, y_true: np.ndarray, probs: np.ndarray) -> "ProbabilityCalibrator":
        y_arr = np.asarray(y_true)
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
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Calibrator has not been fitted")
        p_arr = np.asarray(probs)
        if self.method == "platt":
            return self.model.predict_proba(p_arr.reshape(-1, 1))[:, 1]
        return self.model.predict(p_arr)


class CalibratedModel:
    """Wrap a model and apply probability calibration on predict_proba."""

    def __init__(self, model: Any, calibrator: ProbabilityCalibrator) -> None:
        self.model = model
        self.calibrator = calibrator

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - delegation
        return getattr(self.model, item)

    def predict_proba(self, X: Any) -> np.ndarray:
        raw = self.model.predict_proba(X)
        if raw.ndim == 2 and raw.shape[1] == 2:
            calibrated = self.calibrator.predict(raw[:, 1])
            return np.column_stack([1 - calibrated, calibrated])
        return self.calibrator.predict(raw)


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

