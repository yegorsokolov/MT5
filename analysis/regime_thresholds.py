from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


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
    y = np.asarray(y_true)
    p = np.asarray(probs)
    r = np.asarray(regimes)

    thresholds: Dict[int, float] = {}
    preds = np.zeros_like(y, dtype=int)

    for regime in np.unique(r):
        mask = r == regime
        if mask.sum() == 0:
            continue
        probs_reg = p[mask]
        y_reg = y[mask]
        unique_thr = np.unique(probs_reg)
        best_thr = 0.5
        best_f1 = -1.0
        for thr in unique_thr:
            pred = probs_reg > thr
            tp = np.sum((y_reg == 1) & pred)
            fp = np.sum((y_reg == 0) & pred)
            fn = np.sum((y_reg == 1) & ~pred)
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        thresholds[int(regime)] = float(best_thr)
        preds[mask] = (probs_reg > best_thr).astype(int)

    return thresholds, preds
