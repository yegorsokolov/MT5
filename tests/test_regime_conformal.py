import numpy as np
from pathlib import Path
import sys
import types

sys.path.append(str(Path(__file__).resolve().parents[1]))

sys.modules.setdefault(
    "telemetry",
    types.SimpleNamespace(
        get_meter=lambda name: types.SimpleNamespace(
            create_histogram=lambda *a, **k: types.SimpleNamespace(
                record=lambda *a, **k: None
            )
        )
    ),
)

from models import conformal


def test_regime_specific_intervals():
    rng = np.random.default_rng(0)
    n_calib = 1000
    # Calibration data with two regimes of different noise levels
    X_calib = rng.normal(size=n_calib * 2)
    regimes_calib = np.repeat([0, 1], n_calib)
    noise_scale = np.where(regimes_calib == 0, 0.5, 1.5)
    y_calib = X_calib + rng.normal(scale=noise_scale)
    preds_calib = X_calib
    residuals = y_calib - preds_calib

    q_global = conformal.fit_residuals(residuals, alpha=0.1)
    q_reg = conformal.fit_residuals(residuals, alpha=0.1, regimes=regimes_calib)

    # Test data
    X_test = rng.normal(size=n_calib * 2)
    regimes_test = np.repeat([0, 1], n_calib)
    noise_scale_test = np.where(regimes_test == 0, 0.5, 1.5)
    y_test = X_test + rng.normal(scale=noise_scale_test)
    preds_test = X_test

    lower_g, upper_g = conformal.predict_interval(preds_test, q_global)
    lower_r, upper_r = conformal.predict_interval(preds_test, q_reg, regimes_test)

    cov_g = [
        conformal.evaluate_coverage(
            y_test[regimes_test == r],
            lower_g[regimes_test == r],
            upper_g[regimes_test == r],
        )
        for r in (0, 1)
    ]
    cov_r = [
        conformal.evaluate_coverage(
            y_test[regimes_test == r],
            lower_r[regimes_test == r],
            upper_r[regimes_test == r],
        )
        for r in (0, 1)
    ]
    diff_g = [abs(c - 0.9) for c in cov_g]
    diff_r = [abs(c - 0.9) for c in cov_r]
    assert max(diff_r) < max(diff_g)
