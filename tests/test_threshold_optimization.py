import numpy as np


def test_threshold_optimization_improves_f1():
    """Best threshold should improve F1 over default 0.5 when different."""
    y_true = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.4, 0.35, 0.8])

    def f1_score(y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        if tp == 0 and (fp > 0 or fn > 0):
            return 0.0
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    thresholds = np.unique(probs)
    f1_default = f1_score(y_true, (probs > 0.5).astype(int))
    best_thr, f1_best = 0.5, f1_default
    for thr in thresholds:
        f1_val = f1_score(y_true, (probs > thr).astype(int))
        if f1_val > f1_best:
            best_thr, f1_best = thr, f1_val

    assert abs(best_thr - 0.5) > 1e-6
    assert f1_best > f1_default
