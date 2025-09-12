from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
if "scipy" in sys.modules:
    del sys.modules["scipy"]
if "scipy.stats" in sys.modules:
    del sys.modules["scipy.stats"]
import scipy  # noqa: F401
from analysis.evaluate import risk_adjusted_metrics


def _search_threshold(probs: np.ndarray, y_true: np.ndarray, metric: str) -> float:
    unique = np.unique(probs)
    best_thr = 0.5
    best_val = -np.inf
    for thr in unique:
        preds = (probs >= thr).astype(int)
        if metric == "f1":
            tp = np.sum((y_true == 1) & (preds == 1))
            fp = np.sum((y_true == 0) & (preds == 1))
            fn = np.sum((y_true == 1) & (preds == 0))
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            val = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )
        else:
            val = risk_adjusted_metrics(y_true, preds)[metric]
        if val > best_val:
            best_val = val
            best_thr = float(thr)
    return best_thr


def test_sharpe_threshold_outperforms_f1():
    """Sharpe-optimised threshold yields higher Sharpe ratio than F1 threshold."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0])
    probs = np.array([0.55, 0.6, 0.65, 0.9, 0.8, 0.82, 0.84])

    f1_thr = _search_threshold(probs, y_true, "f1")
    sharpe_thr = _search_threshold(probs, y_true, "sharpe")

    sharpe_f1 = risk_adjusted_metrics(y_true, (probs >= f1_thr).astype(int))["sharpe"]
    sharpe_best = risk_adjusted_metrics(
        y_true, (probs >= sharpe_thr).astype(int)
    )["sharpe"]

    assert sharpe_best > sharpe_f1
