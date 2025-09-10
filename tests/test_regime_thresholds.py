import numpy as np
from analysis.regime_thresholds import find_regime_thresholds
from analysis.prob_calibration import ProbabilityCalibrator
from sklearn.metrics import brier_score_loss


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


def test_calibration_and_thresholds_improve_metrics():
    np.random.seed(1)
    n = 200
    regimes = np.array([0] * 100 + [1] * 100)
    y_true = np.array([0] * 50 + [1] * 50 + [0] * 50 + [1] * 50)
    probs_true = np.zeros(n)
    probs_true[:50] = 0.3
    probs_true[50:100] = 0.7
    probs_true[100:150] = 0.6
    probs_true[150:] = 0.8
    probs_true += np.random.normal(0, 0.03, size=n)
    probs_true = np.clip(probs_true, 0, 1)
    probs_uncal = probs_true ** 2

    brier_raw = brier_score_loss(y_true, probs_uncal)
    cal = ProbabilityCalibrator(method="platt").fit(
        y_true, probs_uncal, regimes=regimes
    )
    probs_cal = cal.predict(probs_uncal)
    brier_cal = brier_score_loss(y_true, probs_cal)
    assert brier_cal < brier_raw

    preds_default = (probs_cal > 0.5).astype(int)
    preds_regime = cal.predict_classes(probs_uncal, regimes=regimes)
    f1_default = f1_score(y_true, preds_default)
    f1_regime = f1_score(y_true, preds_regime)
    assert f1_regime > f1_default
    assert len(cal.regime_thresholds) == 2
    assert cal.regime_thresholds[0] != cal.regime_thresholds[1]
