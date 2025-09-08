import numpy as np
from analysis.regime_thresholds import find_regime_thresholds


def f1_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def test_regime_thresholds_distinct_and_better_f1():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    probs = np.array([0.2, 0.6, 0.8, 0.9, 0.4, 0.5, 0.6, 0.7])
    regimes = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    thr, preds = find_regime_thresholds(y_true, probs, regimes)
    assert len(thr) == 2
    assert thr[0] != thr[1]

    f1_default = f1_score(y_true, (probs > 0.5).astype(int))
    f1_regime = f1_score(y_true, preds)
    assert f1_regime > f1_default
