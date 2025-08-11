import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "price_distribution", ROOT / "models" / "price_distribution.py"
)
price_mod = importlib.util.module_from_spec(spec)
sys.modules["price_distribution"] = price_mod
spec.loader.exec_module(price_mod)  # type: ignore
PriceDistributionModel = price_mod.PriceDistributionModel

import numpy as np


def test_price_distribution_calibration():
    np.random.seed(0)
    X = np.random.randn(500, 1)
    returns = 0.5 * X[:, 0] + np.random.normal(scale=0.1, size=500)
    model = PriceDistributionModel(input_dim=1, hidden_dim=8, n_components=2)
    model.fit(X, returns, epochs=1000)
    x0 = np.array([[0.0]])
    var_pred = model.percentile(x0, 0.05, n_samples=10000)[0]
    true_var = 0.1 * -1.6448536269514729  # 5% quantile of standard normal
    assert abs(var_pred - true_var) < 0.05
