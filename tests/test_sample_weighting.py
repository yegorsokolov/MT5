import numpy as np
import sys

# Ensure scipy is available for lightgbm during tests
sys.modules.pop("scipy", None)
sys.modules.pop("scipy.stats", None)
sys.modules.pop("scipy.sparse", None)
import scipy

sys.modules["scipy"] = scipy
sys.modules["scipy.stats"] = scipy.stats
sys.modules["scipy.sparse"] = scipy.sparse

from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from lightgbm import LGBMClassifier


def test_combined_weighting_improves_f1_on_imbalance():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 1))
    y = np.zeros(n, dtype=int)
    # Early period: almost all negatives
    y[:150] = (X[:150, 0] > 2).astype(int)
    # Recent period: mostly positives
    y[150:] = (X[150:, 0] > -0.5).astype(int)
    t = np.arange(n)

    clf = LGBMClassifier(n_estimators=20, random_state=0)
    clf.fit(X, y)
    f1_un = f1_score(y[150:], clf.predict(X[150:]))

    cw = compute_sample_weight("balanced", y)
    decay = 0.5 ** ((t.max() - t) / 20)
    sw = cw * decay
    clf_w = LGBMClassifier(n_estimators=20, random_state=0)
    clf_w.fit(X, y, sample_weight=sw)
    f1_w = f1_score(y[150:], clf_w.predict(X[150:]))

    assert f1_w > f1_un
