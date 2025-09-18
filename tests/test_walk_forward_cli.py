import json
import types
import contextlib
import sys
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner


def _prepare_cli(monkeypatch):
    metrics: list[float] = []

    def log_metric(name, value):
        metrics.append(value)

    mlflow_stub = types.SimpleNamespace(
        start_run=lambda *a, **k: contextlib.nullcontext(),
        log_metric=log_metric,
    )
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_stub)

    mlflow_client_stub = types.SimpleNamespace(
        start_run=lambda *a, **k: None,
        end_run=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "analytics.mlflow_client", mlflow_client_stub)
    baseline_stub = types.SimpleNamespace(
        backtest=lambda *a, **k: 0.0,
        run_search=lambda *a, **k: {},
    )
    monkeypatch.setitem(sys.modules, "tuning.baseline_opt", baseline_stub)
    auto_search_stub = types.SimpleNamespace(run_search=lambda *a, **k: ({}, pd.DataFrame()))
    monkeypatch.setitem(sys.modules, "tuning.auto_search", auto_search_stub)
    train_graphnet_stub = types.SimpleNamespace(train_graphnet=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "train_graphnet", train_graphnet_stub)
    tpd_stub = types.SimpleNamespace(
        prepare_features=lambda df: (df, df),
        train_price_distribution=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "train_price_distribution", tpd_stub)
    train_nn_stub = types.SimpleNamespace(main=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "train_nn", train_nn_stub)
    training_pkg = types.ModuleType("training")
    training_pkg.__path__ = []  # type: ignore[attr-defined]
    pipeline_stub = types.ModuleType("training.pipeline")
    pipeline_stub.launch = lambda *a, **k: None
    training_pkg.pipeline = pipeline_stub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "training", training_pkg)
    monkeypatch.setitem(sys.modules, "training.pipeline", pipeline_stub)
    env_module = types.ModuleType("utils.environment")
    env_module.ensure_environment = lambda: None
    monkeypatch.setitem(sys.modules, "utils.environment", env_module)

    utils_module = types.ModuleType("utils")
    utils_module.__path__ = []  # type: ignore[attr-defined]

    def fake_load_config(*_a, **_k):
        return types.SimpleNamespace(model_dump=lambda: {})

    utils_module.load_config = fake_load_config  # type: ignore[attr-defined]
    utils_module.environment = env_module  # type: ignore[attr-defined]
    utils_module.ensure_environment = env_module.ensure_environment  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "utils", utils_module)
    backtest_stub = types.SimpleNamespace(run_rolling_backtest=lambda *a, **k: {})
    monkeypatch.setitem(sys.modules, "backtest", backtest_stub)
    sys.path.append(str(Path(__file__).resolve().parents[1]))

    from train_cli import app

    return app, metrics


def test_walk_forward_cli_no_leakage(tmp_path, monkeypatch):
    df = pd.DataFrame({"return": [0.0, 1.0, 2.0, 3.0, 4.0]})
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    app, metrics = _prepare_cli(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "walk-forward",
            "--data",
            str(data_path),
            "--window-length",
            "3",
            "--step-size",
            "1",
            "--model-type",
            "mean",
        ],
    )

    assert result.exit_code == 0
    records = json.loads(result.stdout.strip())

    assert len(records) == 2
    assert len(metrics) == 2

    for rec in records:
        assert rec["train_end"] < rec["test_start"]

    assert all(abs(rec["rmse"] - 2.0) < 1e-9 for rec in records)
