import importlib
import logging
import sys
import types
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _create_stub_module(name: str, attrs: dict[str, object]) -> types.ModuleType:
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


def _write_returns(tmp_path: Path) -> Path:
    data = pd.DataFrame(
        {
            "return": [
                0.01,
                -0.02,
                0.015,
                -0.005,
                0.02,
                -0.01,
            ]
        }
    )
    path = tmp_path / "returns.csv"
    data.to_csv(path, index=False)
    return path


def test_walk_forward_train_without_mlflow(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.WARNING)

    monkeypatch.syspath_prepend(str(PROJECT_ROOT))

    monkeypatch.setitem(sys.modules, "mlflow", None)

    utils_stub = _create_stub_module("utils", {"load_config": lambda *args, **kwargs: {}})
    monkeypatch.setitem(sys.modules, "utils", utils_stub)

    def _log_exceptions(func):
        return func

    log_utils_stub = _create_stub_module(
        "log_utils",
        {
            "setup_logging": lambda *args, **kwargs: None,
            "log_exceptions": _log_exceptions,
        },
    )
    monkeypatch.setitem(sys.modules, "log_utils", log_utils_stub)

    backtest_stub = _create_stub_module(
        "backtest", {"run_rolling_backtest": lambda *args, **kwargs: {}}
    )
    monkeypatch.setitem(sys.modules, "backtest", backtest_stub)

    data_path = _write_returns(tmp_path)

    walk_forward = importlib.import_module("walk_forward")
    try:
        result = walk_forward.walk_forward_train(data_path, window_length=3, step_size=2)
    finally:
        sys.modules.pop("walk_forward", None)

    assert not result.empty
    assert "rmse" in result.columns
    assert "skipping tracking" in caplog.text
