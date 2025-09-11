import sys
import numpy as np

# Remove scipy stubs inserted by tests.conftest so real scipy/sklearn can be used
sys.modules.pop("scipy", None)
sys.modules.pop("scipy.stats", None)
from sklearn.metrics import f1_score, precision_recall_curve


def test_threshold_optimization_improves_f1():
    """Best threshold should improve F1 over default 0.5 when different."""
    y_true = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.4, 0.35, 0.8])

    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1[:-1]))
    best_thr = thresholds[best_idx]
    f1_best = f1_score(y_true, (probs >= best_thr).astype(int))
    f1_default = f1_score(y_true, (probs >= 0.5).astype(int))

    assert abs(best_thr - 0.5) > 1e-6
    assert f1_best > f1_default
