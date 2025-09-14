import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure real scipy is available for statsmodels
import importlib

for name in ["scipy", "scipy.stats"]:
    sys.modules.pop(name, None)
import scipy  # noqa: F401  - imported for side effect
import scipy.stats  # noqa: F401

from models.arima import ARIMAModel


def test_arima_forecast_shape():
    series = np.sin(np.linspace(0, 4, 50))
    model = ARIMAModel(order=(2, 0, 0))
    model.fit(series)
    forecast = model.forecast(steps=3)
    assert forecast.shape == (3,)
