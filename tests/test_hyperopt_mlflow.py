"""Tests for MLflow handling in tuning hyperparameter utilities."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

optuna_stub = types.ModuleType("optuna")
optuna_stub.create_study = lambda *a, **k: None
optuna_stub.Trial = type("Trial", (), {})
sys.modules.setdefault("optuna", optuna_stub)


class _BaseModel(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def model_dump(self) -> dict:
        return dict(self)


class _AppConfig(_BaseModel):
    pass


pydantic_stub = types.ModuleType("pydantic")
pydantic_stub.BaseModel = _BaseModel
pydantic_stub.ValidationError = type("ValidationError", (Exception,), {})
sys.modules.setdefault("pydantic", pydantic_stub)

config_models_stub = types.ModuleType("config_models")
config_models_stub.AppConfig = _AppConfig
config_models_stub.ConfigError = type("ConfigError", (Exception,), {})
sys.modules.setdefault("config_models", config_models_stub)

filelock_stub = types.ModuleType("filelock")


class _DummyFileLock:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


filelock_stub.FileLock = _DummyFileLock
sys.modules.setdefault("filelock", filelock_stub)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tuning import hyperopt


@pytest.mark.parametrize(
    ("func_name", "module_name", "entrypoint"),
    [
        ("tune_lgbm", "train", "main"),
        ("tune_transformer", "train_nn", "launch"),
        ("tune_rl", "train_rl", "launch"),
    ],
)
def test_tune_functions_end_run_on_optimize_failure(
    monkeypatch: pytest.MonkeyPatch, func_name: str, module_name: str, entrypoint: str
) -> None:
    """Ensure MLflow runs are closed even when Optuna raises during optimization."""

    dummy_module = types.ModuleType(module_name)
    setattr(dummy_module, entrypoint, lambda cfg: 0.0)
    monkeypatch.setitem(sys.modules, module_name, dummy_module)

    mlflow_calls = {"start": 0, "end": 0}

    def fake_start(run_name: str, cfg: dict) -> None:
        mlflow_calls["start"] += 1

    def fake_end() -> None:
        mlflow_calls["end"] += 1

    monkeypatch.setattr(hyperopt.mlflow, "start_run", fake_start)
    monkeypatch.setattr(hyperopt.mlflow, "end_run", fake_end)
    monkeypatch.setattr(hyperopt.mlflow, "log_params", lambda params: None)
    monkeypatch.setattr(hyperopt.mlflow, "log_metric", lambda key, value: None)

    class FaultyStudy:
        best_params: dict = {}
        best_value: float = 0.0

        def optimize(self, *args, **kwargs) -> None:  # noqa: ANN003
            raise RuntimeError("optimize failed")

    monkeypatch.setattr(hyperopt, "_study", lambda storage, name: FaultyStudy())

    tune = getattr(hyperopt, func_name)

    with pytest.raises(RuntimeError, match="optimize failed"):
        tune({}, n_trials=1)

    assert mlflow_calls["start"] == 1
    assert mlflow_calls["end"] == 1
