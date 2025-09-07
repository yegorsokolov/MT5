import math
from pathlib import Path
import sys

import numpy as np

# ensure repository root is on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.gp_calibrator import GPCalibrator


def _norm_cdf(x):
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def test_gp_calibrator_coverage_and_brier():
    rng = np.random.default_rng(0)
    n_train = 200
    X_train = rng.normal(size=n_train)
    noise_std_train = 0.1 + 0.5 * np.abs(X_train)
    y_train = X_train + rng.normal(scale=noise_std_train)
    preds_train = X_train

    calib = GPCalibrator().fit(preds_train, y_train)

    n_test = 400
    X_test = rng.normal(size=n_test)
    noise_std_test = 0.1 + 0.5 * np.abs(X_test)
    y_test = X_test + rng.normal(scale=noise_std_test)
    preds_test = X_test

    var = calib.predict_variance(preds_test)
    lower, upper = calib.predict_interval(preds_test)
    coverage = np.mean((y_test >= lower) & (y_test <= upper))
    assert abs(coverage - 0.95) < 0.05

    baseline_var = np.var(y_train - preds_train)
    p_base = _norm_cdf((preds_test - 0) / math.sqrt(baseline_var))
    p_cal = _norm_cdf((preds_test - 0) / np.sqrt(var))
    y_bin = (y_test > 0).astype(float)
    brier_base = np.mean((p_base - y_bin) ** 2)
    brier_cal = np.mean((p_cal - y_bin) ** 2)
    assert brier_cal < brier_base
