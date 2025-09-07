import numpy as np
import sys

sys.modules.pop("scipy", None)
sys.modules.pop("scipy.stats", None)
sys.modules.pop("scipy.sparse", None)
import scipy

sys.modules["scipy"] = scipy
sys.modules["scipy.stats"] = scipy.stats
sys.modules["scipy.sparse"] = scipy.sparse
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier


def test_balance_classes_applies_weights():
    rng = np.random.default_rng(0)
    n = 200
    y = (rng.random(n) > 0.9).astype(int)
    X = y[:, None] + rng.normal(scale=0.1, size=(n, 1))

    clf = LGBMClassifier(n_estimators=10, random_state=0)
    clf.fit(X, y)
    ratio_un = clf.predict(X).mean()

    w = compute_sample_weight("balanced", y)
    clf_w = LGBMClassifier(n_estimators=10, random_state=0)
    clf_w.fit(X, y, sample_weight=w)
    ratio_w = clf_w.predict(X).mean()

    assert ratio_w > ratio_un


def test_time_decay_weights_increase_over_time():
    half_life = 5
    t = np.arange(10)
    t_max = t.max()
    w = 0.5 ** ((t_max - t) / half_life)
    assert np.all(np.diff(w) > 0)


def test_time_decay_weighting_improves_f1_on_drift():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 1))
    y = np.zeros(n, dtype=int)
    split = n // 2
    y[:split] = (X[:split, 0] > 0).astype(int)
    y[split:] = (X[split:, 0] < 0).astype(int)

    clf = LGBMClassifier(n_estimators=20, random_state=0)
    clf.fit(X, y)
    f1_un = f1_score(y[split:], clf.predict(X[split:]))

    t = np.arange(n)
    w = 0.5 ** ((t.max() - t) / 20)
    clf_w = LGBMClassifier(n_estimators=20, random_state=0)
    clf_w.fit(X, y, sample_weight=w)
    f1_w = f1_score(y[split:], clf_w.predict(X[split:]))

    assert f1_w > f1_un
