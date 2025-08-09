import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

sklearn = pytest.importorskip("sklearn")
shap = pytest.importorskip("shap")
from sklearn.ensemble import RandomForestClassifier

from analysis.interpret_model import compute_shap_importance


def test_shap_identifies_top_feature():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    y = (X["a"] + 0.1 * rng.normal(size=200) > 0).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    fi = compute_shap_importance(model, X)
    assert fi.iloc[0]["feature"] == "a"
