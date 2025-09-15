import numpy as np
import pytest
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


def test_calibrate_intervals_serialisation_roundtrip():
    rng = np.random.default_rng(1)
    n_val = 800
    n_test = 200
    x_val = rng.normal(size=n_val)
    noise_val = rng.normal(scale=0.4, size=n_val)
    y_val = x_val + noise_val
    preds_val = x_val
    params, residuals, (lower_val, upper_val) = conformal.calibrate_intervals(
        y_val,
        preds_val,
        alpha=0.1,
    )
    assert isinstance(params, conformal.ConformalIntervalParams)
    assert residuals.shape == y_val.shape
    observed_cov = conformal.evaluate_coverage(y_val, lower_val, upper_val)
    assert params.coverage == pytest.approx(observed_cov)

    x_test = rng.normal(size=n_test)
    noise_test = rng.normal(scale=0.4, size=n_test)
    y_test = x_test + noise_test
    preds_test = x_test
    lower, upper = params.predict_interval(preds_test)
    cov = conformal.evaluate_coverage(y_test, lower, upper)
    assert abs(cov - (1 - params.alpha)) < 0.05

    restored = conformal.ConformalIntervalParams.from_dict(params.to_dict())
    assert restored.alpha == pytest.approx(params.alpha)
    if isinstance(params.quantiles, dict):
        assert isinstance(restored.quantiles, dict)
        for key, value in params.quantiles.items():
            assert restored.quantiles[key] == pytest.approx(value)
    else:
        assert restored.quantiles == pytest.approx(params.quantiles)
    assert restored.coverage == pytest.approx(params.coverage)


def test_interval_params_regime_serialisation():
    params = conformal.ConformalIntervalParams(
        alpha=0.2,
        quantiles={0: 0.15, 1: 0.25},
        coverage=0.88,
        coverage_by_regime={0: 0.9, 1: 0.86},
    )
    restored = conformal.ConformalIntervalParams.from_dict(params.to_dict())
    assert restored.alpha == pytest.approx(params.alpha)
    assert restored.coverage == pytest.approx(params.coverage)
    assert isinstance(restored.coverage_by_regime, dict)
    for key, value in params.coverage_by_regime.items():
        assert restored.coverage_by_regime[key] == pytest.approx(value)
    assert isinstance(restored.quantiles, dict)
    for key, value in params.quantiles.items():
        assert restored.quantiles[key] == pytest.approx(value)
