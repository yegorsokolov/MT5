import numpy as np
import pandas as pd
import sys

# Ensure scipy is available for scikit-learn
sys.modules.pop("scipy", None)
sys.modules.pop("scipy.stats", None)
sys.modules.pop("scipy.sparse", None)
import scipy
sys.modules["scipy"] = scipy
sys.modules["scipy.stats"] = scipy.stats
sys.modules["scipy.sparse"] = scipy.sparse

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

from analysis.data_quality import score_samples


def test_score_samples_downweights_noisy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(10, 3)))
    X.loc[0] += 10
    w = score_samples(X)
    assert w[0] < w[1:].mean()


def test_data_quality_weight_improves_validation_accuracy():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 1))
    y = (X[:, 0] > 0).astype(int)
    X[:40] += 10
    y[:40] = 1 - y[:40]
    X_tr, X_val = X[:150], X[150:]
    y_tr, y_val = y[:150], y[150:]
    dq_w = score_samples(pd.DataFrame(X_tr))
    cw = compute_sample_weight("balanced", y_tr)
    t = np.arange(n)
    decay = 0.5 ** ((t.max() - t[:150]) / 20)
    sw = cw * decay * dq_w
    clf_un = LogisticRegression(max_iter=1000)
    clf_un.fit(X_tr, y_tr)
    clf_w = LogisticRegression(max_iter=1000)
    clf_w.fit(X_tr, y_tr, sample_weight=sw)
    f1_un = f1_score(y_val, clf_un.predict(X_val))
    f1_w = f1_score(y_val, clf_w.predict(X_val))
    assert f1_w > f1_un
