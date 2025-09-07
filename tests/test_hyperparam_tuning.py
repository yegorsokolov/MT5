import numpy as np
import pandas as pd
import sys
from pathlib import Path
import types

sys.path.append(str(Path(__file__).resolve().parents[1]))

mlflow_stub = types.SimpleNamespace(
    log_params=lambda *a, **k: None,
    log_metric=lambda *a, **k: None,
)
sys.modules.setdefault("analytics.mlflow_client", mlflow_stub)

from tuning import tuning as tuning_module


def test_tune_lightgbm_executes_trials(monkeypatch):
    calls = []

    def train_fn(params, trial):  # noqa: ARG001
        calls.append(params)
        return params["num_leaves"]

    monkeypatch.setattr(tuning_module.mlflow, "log_params", lambda *a, **k: None)
    monkeypatch.setattr(tuning_module.mlflow, "log_metric", lambda *a, **k: None)

    best = tuning_module.tune_lightgbm(train_fn, n_trials=3)
    assert len(calls) == 3
    assert set(["num_leaves", "learning_rate", "max_depth"]).issubset(best)


