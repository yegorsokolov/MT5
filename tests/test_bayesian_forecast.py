import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.bayesian_forecast import BayesianForecaster


def test_calibration_and_logging():
    np.random.seed(0)
    true_mu = 0.01
    true_sigma = 0.02
    data = np.random.normal(true_mu, true_sigma, size=200)

    forecaster = BayesianForecaster()
    forecaster.fit(data)
    pred = forecaster.forecast(cred_level=0.9)

    samples = np.random.normal(true_mu, true_sigma, size=1000)
    coverage = (samples >= pred["lower"]) & (samples <= pred["upper"])
    rate = coverage.mean()
    assert abs(rate - 0.9) < 0.1

    report_file = Path("reports/bayesian_forecasts/posterior_diagnostics.csv")
    assert report_file.exists()
    content = report_file.read_text().strip().splitlines()
    assert content, "log file should not be empty"
