import numpy as np
import sys
from pathlib import Path

# Ensure repository root on path and remove scipy stubs
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.pop("scipy", None)
sys.modules.pop("scipy.stats", None)
import scipy  # noqa: F401  # ensure real scipy is available
from lightgbm import LGBMClassifier
from mt5.focal_loss import make_focal_loss, make_focal_loss_metric
from sklearn.datasets import make_classification
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split


def test_focal_loss_improves_recall():
    X, y = make_classification(
        n_samples=1000, n_features=10, weights=[0.98, 0.02], flip_y=0, random_state=0
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )

    clf = LGBMClassifier(n_estimators=20, random_state=0)
    clf.fit(X_train, y_train)
    base_recall = recall_score(y_test, clf.predict(X_test))

    clf_focal = LGBMClassifier(
        n_estimators=20, random_state=0, objective=make_focal_loss()
    )
    clf_focal.fit(X_train, y_train, eval_metric=make_focal_loss_metric())
    raw = clf_focal.predict(X_test)
    prob = 1.0 / (1.0 + np.exp(-raw))
    focal_recall = recall_score(y_test, (prob > 0.5).astype(int))

    assert focal_recall >= base_recall
