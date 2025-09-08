import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from analysis.prob_calibration import ProbabilityCalibrator


def test_cv_calibration_improves_brier():
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10, random_state=0
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=1
    )
    base = RandomForestClassifier(n_estimators=25, random_state=0)
    base.fit(X_train, y_train)
    probs = base.predict_proba(X_test)[:, 1]
    brier_raw = brier_score_loss(y_test, probs)

    calibrator = ProbabilityCalibrator(method="platt", cv=3).fit(
        y_train, base_model=RandomForestClassifier(n_estimators=25, random_state=0), X=X_train
    )
    calibrated_model = calibrator.model
    probs_cal = calibrated_model.predict_proba(X_test)[:, 1]
    brier_cal = brier_score_loss(y_test, probs_cal)
    assert brier_cal < brier_raw
