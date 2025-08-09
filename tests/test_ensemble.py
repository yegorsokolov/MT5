import numpy as np
import pandas as pd
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ensemble", Path(__file__).resolve().parents[1] / "models" / "ensemble.py"
)
ensemble_mod = importlib.util.module_from_spec(spec)
sys.modules["ensemble"] = ensemble_mod
spec.loader.exec_module(ensemble_mod)
EnsembleModel = ensemble_mod.EnsembleModel


class DummyModel:
    def __init__(self, probs):
        self.probs = np.asarray(probs)

    def predict_proba(self, X):  # pragma: no cover - simple stub
        return np.column_stack([1 - self.probs, self.probs])


def test_weighted_average():
    df = pd.DataFrame({"x": [1, 2]})
    m1 = DummyModel([0.2, 0.4])
    m2 = DummyModel([0.6, 0.8])
    ens = EnsembleModel({"m1": m1, "m2": m2}, weights={"m1": 0.25, "m2": 0.75})
    preds = ens.predict(df)
    assert np.allclose(preds["m1"], [0.2, 0.4])
    assert np.allclose(preds["m2"], [0.6, 0.8])
    expected = 0.25 * np.array([0.2, 0.4]) + 0.75 * np.array([0.6, 0.8])
    assert np.allclose(preds["ensemble"], expected)
