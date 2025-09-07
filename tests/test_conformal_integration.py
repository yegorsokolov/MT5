import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models import conformal


def test_conformal_interval_coverage():
    rng = np.random.default_rng(1)
    X = rng.normal(size=2000)
    noise = rng.normal(scale=0.5, size=2000)
    y = X + noise
    preds_calib = X[:1000]
    y_calib = y[:1000]
    q = conformal.fit_residuals(y_calib - preds_calib, alpha=0.1)
    preds_test = X[1000:]
    y_test = y[1000:]
    lower, upper = conformal.predict_interval(preds_test, q)
    cov = conformal.evaluate_coverage(y_test, lower, upper)
    assert abs(cov - 0.9) < 0.05
