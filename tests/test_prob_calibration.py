import numpy as np
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from analysis.prob_calibration import ProbabilityCalibrator


def test_platt_calibration():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    cal = ProbabilityCalibrator(method="platt").fit(y, p)
    calibrated = cal.predict(p)
    assert calibrated.shape == p.shape
    assert np.all(calibrated >= 0) and np.all(calibrated <= 1)

def test_isotonic_calibration():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.2, 0.3, 0.7, 0.8])
    cal = ProbabilityCalibrator(method="isotonic").fit(y, p)
    calibrated = cal.predict(p)
    assert calibrated.shape == p.shape
    assert np.all(calibrated >= 0) and np.all(calibrated <= 1)
