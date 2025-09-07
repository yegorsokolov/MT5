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
