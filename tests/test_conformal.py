import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models import conformal


def test_conformal_calibration():
    rng = np.random.default_rng(0)
    n_calib = 1000
    n_test = 200
    X_calib = rng.normal(size=n_calib)
    noise_calib = rng.normal(scale=0.5, size=n_calib)
    y_calib = X_calib + noise_calib
    preds_calib = X_calib  # model predicts X perfectly
    residuals = np.abs(preds_calib - y_calib)
    q = conformal.fit_residuals(residuals, alpha=0.1)
    X_test = rng.normal(size=n_test)
    noise_test = rng.normal(scale=0.5, size=n_test)
    y_test = X_test + noise_test
    preds_test = X_test
    lower, upper = conformal.predict_interval(preds_test, q)
    cov = conformal.evaluate_coverage(y_test, lower, upper)
    assert abs(cov - 0.9) < 0.05
