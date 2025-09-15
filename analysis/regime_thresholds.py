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
    try:
        from sklearn.metrics import precision_recall_curve  # pylint: disable=import-outside-toplevel
    except (ModuleNotFoundError, ImportError):
        import sys

        sys.modules.pop("scipy", None)
        sys.modules.pop("scipy.stats", None)
        from sklearn.metrics import precision_recall_curve  # type: ignore  # pylint: disable=import-outside-toplevel

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
