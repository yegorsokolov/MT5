import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.scenario_diffusion import ScenarioDiffusion


def _knn_predict(X_train, y_train, X_test, k=3):
    preds = []
    for x in X_test:
        dists = np.abs(X_train.flatten() - x[0])
        idx = np.argsort(dists)[:k]
        vote = int(round(y_train[idx].mean()))
        preds.append(vote)
    return np.array(preds)


def _recall(y_true, y_pred, cls):
    tp = np.sum((y_true == cls) & (y_pred == cls))
    fn = np.sum((y_true == cls) & (y_pred != cls))
    return tp / (tp + fn) if (tp + fn) else 0.0


def test_diffusion_augmentation_improves_recall():
    np.random.seed(0)
    torch.manual_seed(0)
    X_major = np.random.normal(loc=0.1, scale=0.01, size=(40, 1))
    y_major = np.ones(40, dtype=int)
    X_minor = np.random.normal(loc=0.09, scale=0.005, size=(5, 1))
    y_minor = np.zeros(5, dtype=int)
    X = np.concatenate([X_major, X_minor], axis=0)
    y = np.concatenate([y_major, y_minor], axis=0)
    X_test = np.array([[0.1], [-0.3]])
    y_test = np.array([1, 0])

    base_rec = _recall(y_test, _knn_predict(X, y, X_test), 0)

    model = ScenarioDiffusion(seq_len=20)
    crash_samples = []
    for _ in range(20):
        path = model.sample_crash_recovery(20)
        path = np.clip(path, -0.3, 0.3)
        crash_samples.append(path.min())
    X_aug = np.concatenate([X, np.array(crash_samples).reshape(-1, 1)], axis=0)
    y_aug = np.concatenate([y, np.zeros(len(crash_samples), dtype=int)], axis=0)

    aug_rec = _recall(y_test, _knn_predict(X_aug, y_aug, X_test), 0)

    assert aug_rec > base_rec
