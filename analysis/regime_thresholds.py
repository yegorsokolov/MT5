from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def _precision_recall_curve_numpy(
    y_true: np.ndarray, probs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a precision/recall curve using only NumPy operations."""

    if y_true.size == 0:
        return (
            np.array([1.0], dtype=float),
            np.array([0.0], dtype=float),
            np.array([], dtype=float),
        )

    order = np.argsort(probs, kind="mergesort")[::-1]
    y_sorted = y_true[order]
    probs_sorted = probs[order]

    distinct_value_indices = np.where(np.diff(probs_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_sorted.size - 1]

    true_positives = np.cumsum(y_sorted, dtype=float)[threshold_idxs]
    false_positives = 1 + threshold_idxs - true_positives

    predicted_positive = true_positives + false_positives
    precision = np.zeros_like(true_positives, dtype=float)
    np.divide(true_positives, predicted_positive, out=precision, where=predicted_positive != 0)

    if true_positives[-1] == 0:
        recall = np.ones_like(true_positives, dtype=float)
    else:
        recall = true_positives / true_positives[-1]

    sl = slice(None, None, -1)
    precision = np.hstack((precision[sl], np.array([1.0], dtype=float)))
    recall = np.hstack((recall[sl], np.array([0.0], dtype=float)))
    thresholds = probs_sorted[threshold_idxs][sl]

    return precision, recall, thresholds


try:  # pragma: no cover - exercised via tests when SciPy is available
    from sklearn.metrics import precision_recall_curve  # pylint: disable=import-outside-toplevel
except (ModuleNotFoundError, ImportError):

    def precision_recall_curve(
        y_true: Iterable[int] | np.ndarray,
        probs: Iterable[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_arr = np.asarray(y_true, dtype=int).ravel()
        p_arr = np.asarray(probs, dtype=float).ravel()
        return _precision_recall_curve_numpy(y_arr, p_arr)


def find_regime_thresholds(
    y_true: Iterable[int] | np.ndarray,
    probs: Iterable[float] | np.ndarray,
    regimes: Iterable[int] | np.ndarray,
) -> Tuple[Dict[int, float], np.ndarray]:
    """Return F1-optimal thresholds per regime and resulting predictions.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels.
    probs : array-like
        Predicted probabilities for the positive class.
    regimes : array-like
        Market regime identifier per sample.

    Returns
    -------
    dict
        Mapping of regime to optimal threshold.
    np.ndarray
        Binary predictions generated using the regime-specific thresholds.
    """
    y = np.ravel(np.asarray(y_true))
    p = np.asarray(probs)
    r = np.asarray(regimes)

    if not (len(y) == len(p) == len(r)):
        raise ValueError("Inputs must have the same length")

    thresholds: Dict[int, float] = {}
    preds = np.zeros_like(y, dtype=int)

    for regime in np.unique(r):
        mask = r == regime
        if not np.any(mask):
            continue
        probs_reg = p[mask]
        y_reg = y[mask]
        if probs_reg.size == 0:
            continue

        precision, recall, thr = precision_recall_curve(y_reg, probs_reg)
        if thr.size > 0:
            precision = np.nan_to_num(precision[:-1])
            recall = np.nan_to_num(recall[:-1])
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            best_idx = int(np.nanargmax(f1))
            best_thr = float(thr[best_idx])
        else:
            best_thr = 0.5

        thresholds[int(regime)] = float(best_thr)
        preds[mask] = (probs_reg >= best_thr).astype(int)

    return thresholds, preds
