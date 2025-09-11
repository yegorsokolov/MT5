import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

sklearn = pytest.importorskip("sklearn")
shap = pytest.importorskip("shap")
plt = pytest.importorskip("matplotlib.pyplot")
from sklearn.ensemble import RandomForestClassifier

from analysis.interpret_model import generate_shap_report


def test_generate_shap_report_produces_artifacts(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    y = (X["a"] + 0.1 * rng.normal(size=200) > 0).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    paths = generate_shap_report(model, X, tmp_path)
    assert paths["plot"].exists()
    assert paths["features"].exists()
    fi = pd.read_csv(paths["features"])
    assert fi.iloc[0]["feature"] == "a"
