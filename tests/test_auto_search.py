import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import types

mlflow_stub = types.SimpleNamespace(
    log_params=lambda *a, **k: None,
    log_metric=lambda *a, **k: None,
    log_param=lambda *a, **k: None,
)
sys.modules.setdefault("analytics.mlflow_client", mlflow_stub)

# Stub heavy model modules to avoid optional dependencies during import
sys.modules.setdefault("models.multi_head", types.SimpleNamespace(MultiHeadTransformer=object))
sys.modules.setdefault(
    "models.cross_asset_transformer", types.SimpleNamespace(CrossAssetTransformer=object)
)
class _DummyLGBM:
    def __init__(self, **params):
        self._mean = 0.0

    def fit(self, X, y):  # noqa: D401
        self._mean = float(np.mean(y))

    def predict(self, X):  # noqa: D401
        return np.full(len(X), self._mean)

sys.modules.setdefault("lightgbm", types.SimpleNamespace(LGBMRegressor=_DummyLGBM))

from tuning.auto_search import run_search
from analytics import mlflow_client as mlflow


def test_auto_search_selects_model_and_logs(monkeypatch):
    logs = {"params": [], "metrics": []}

    def log_params(p):
        logs["params"].append(p)

    def log_metric(k, v, step=None):  # noqa: ARG001
        logs["metrics"].append((k, v, step))

    monkeypatch.setattr(mlflow, "log_params", log_params)
    monkeypatch.setattr(mlflow, "log_metric", log_metric)
    monkeypatch.setattr(mlflow, "log_param", lambda *a, **k: None)

    X = np.random.randn(20, 3)
    y = np.random.randn(20)
    best, summary = run_search(X, y, n_trials=1, n_splits=2, model_types=["lightgbm"])

    assert best["model_type"] == "lightgbm"
    assert logs["params"] and logs["metrics"]
    assert len(summary) == 2
