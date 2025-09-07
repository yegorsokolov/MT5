import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models import residual_stacker


def test_residual_stacker_improves_rmse(tmp_path, monkeypatch):
    # ensure model saved to temporary path to avoid polluting repo
    monkeypatch.setattr(
        residual_stacker, "_model_path", lambda name: tmp_path / f"{name}.pkl"
    )

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    coef = np.array([0.4, -0.2, 0.1])
    y = X @ coef + rng.normal(scale=0.1, size=200)

    base_pred = X @ coef + rng.normal(scale=0.5, size=200)

    X_train, X_val = X[:150], X[150:]
    base_train, base_val = base_pred[:150], base_pred[150:]
    y_train, y_val = y[:150], y[150:]

    residual_stacker.train(X_train, base_train, y_train, "demo")
    model = residual_stacker.load("demo")
    assert model is not None
    resid = residual_stacker.predict(X_val, base_val, model)
    final_pred = base_val + resid

    rmse_base = np.sqrt(np.mean((y_val - base_val) ** 2))
    rmse_final = np.sqrt(np.mean((y_val - final_pred) ** 2))
    assert rmse_final < rmse_base
