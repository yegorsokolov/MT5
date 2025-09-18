import contextlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner


def test_ensemble_cli_logs_metrics(tmp_path, monkeypatch):
    metrics: list[tuple[str, float]] = []

    def log_metric(name: str, value: float) -> None:
        metrics.append((name, value))

    mlflow_stub = types.SimpleNamespace(
        start_run=lambda *a, **k: contextlib.nullcontext(),
        log_metric=log_metric,
        log_dict=lambda *a, **k: None,
        end_run=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_stub)

    mlflow_client_stub = types.SimpleNamespace(
        start_run=lambda *a, **k: None,
        end_run=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "analytics.mlflow_client", mlflow_client_stub)

    moe_cfg_holder: dict[str, dict[str, object]] = {}

    class ResourceCapabilities:  # pragma: no cover - simple container
        def __init__(self, cpu_count: int, memory_gb: int, supports_gpu: bool, gpu_count: int = 0) -> None:
            self.cpu_count = cpu_count
            self.memory_gb = memory_gb
            self.supports_gpu = supports_gpu
            self.gpu_count = gpu_count

    def main(cfg=None, data=None):  # pragma: no cover - stub training routine
        log_metric("f1_ensemble", 0.9)
        return {"ensemble": 0.9, "m1": 0.8}

    def train_moe_ensemble(histories, regimes, targets, caps, cfg=None):  # pragma: no cover - stub MOE
        moe_cfg_holder["cfg"] = cfg or {}
        log_metric("mse_mixture", 0.1)
        preds = np.asarray(targets, dtype=float)
        experts = np.tile(preds[:, None], (1, 3))
        return preds, experts

    baseline_stub = types.SimpleNamespace(
        backtest=lambda *a, **k: 0.0,
        run_search=lambda *a, **k: {},
    )
    monkeypatch.setitem(sys.modules, "tuning.baseline_opt", baseline_stub)
    auto_search_stub = types.SimpleNamespace(run_search=lambda *a, **k: ({}, pd.DataFrame()))
    monkeypatch.setitem(sys.modules, "tuning.auto_search", auto_search_stub)
    graph_stub = types.SimpleNamespace(train_graphnet=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "train_graphnet", graph_stub)
    price_stub = types.SimpleNamespace(
        prepare_features=lambda df: (df, df),
        train_price_distribution=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "train_price_distribution", price_stub)
    nn_stub = types.SimpleNamespace(main=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "train_nn", nn_stub)
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

    te_stub = types.SimpleNamespace(
        main=main,
        train_moe_ensemble=train_moe_ensemble,
        ResourceCapabilities=ResourceCapabilities,
    )
    monkeypatch.setitem(sys.modules, "train_ensemble", te_stub)

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from train_cli import app

    setup_called = False
    end_called = False

    def fake_setup(config, experiment=None):
        nonlocal setup_called
        setup_called = True
        return {"ensemble": {"enabled": True}}

    def fake_end():
        nonlocal end_called
        end_called = True

    monkeypatch.setattr("train_cli.setup_training", fake_setup)
    monkeypatch.setattr("train_cli.end_training", fake_end)
    monkeypatch.setattr("train_cli.train_ensemble_main", te_stub.main)
    monkeypatch.setattr("train_cli.train_moe_ensemble", te_stub.train_moe_ensemble)
    monkeypatch.setattr("train_cli.ResourceCapabilities", ResourceCapabilities)

    df = pd.DataFrame(
        {
            "feat_a": [0.1, 0.2, -0.1, -0.2],
            "feat_b": [1.0, 0.5, -0.4, -0.2],
            "signal": [1, 0, 1, 0],
            "regime": [0, 1, 0, 1],
            "history_0": [0.0, 0.1, 0.2, 0.3],
            "history_1": [0.1, 0.2, 0.3, 0.4],
        }
    )
    data_path = tmp_path / "ensemble.csv"
    df.to_csv(data_path, index=False)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "ensemble",
            "--data",
            str(data_path),
            "--target",
            "signal",
            "--feature",
            "feat_a",
            "--feature",
            "feat_b",
            "--moe-regime",
            "regime",
            "--gating-sharpness",
            "7.5",
            "--expert-weight",
            "0.5",
            "--expert-weight",
            "1.5",
            "--expert-weight",
            "2.0",
        ],
    )

    assert result.exit_code == 0
    output_lines = [json.loads(line) for line in result.stdout.strip().splitlines()]
    assert any("ensemble" in line for line in output_lines)
    assert any("mse_mixture" in line for line in output_lines)

    assert setup_called and end_called
    assert metrics and ("f1_ensemble", 0.9) in metrics and ("mse_mixture", 0.1) in metrics
    assert moe_cfg_holder["cfg"]["sharpness"] == 7.5
    assert moe_cfg_holder["cfg"]["expert_weights"] == [0.5, 1.5, 2.0]
