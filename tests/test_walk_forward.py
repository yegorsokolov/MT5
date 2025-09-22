import pandas as pd
import contextlib
import importlib
import logging
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.modules["log_utils"] = types.SimpleNamespace(
    setup_logging=lambda: logging.getLogger(),
    log_exceptions=lambda f: f,
)
sys.modules["backtest"] = types.SimpleNamespace(run_rolling_backtest=lambda cfg: {})
sys.modules["mt5.backtest"] = sys.modules["backtest"]
sys.modules["mlflow"] = types.SimpleNamespace(
    set_experiment=lambda *a, **k: None,
    set_tracking_uri=lambda *a, **k: None,
    start_run=lambda *a, **k: contextlib.nullcontext(),
    log_dict=lambda *a, **k: None,
)

from mt5 import walk_forward

walk_forward.init_logging = lambda: logging.getLogger("test_walk_forward")


def test_aggregate_results():
    results = {
        "XAUUSD": {"avg_sharpe": 1.0, "worst_drawdown": -5.0},
        "GBPUSD": {"avg_sharpe": 0.5, "worst_drawdown": -3.0},
    }
    df = walk_forward.aggregate_results(results)
    assert list(df.columns) == ["symbol", "avg_sharpe", "worst_drawdown"]
    assert len(df) == 2
    assert df[df.symbol == "XAUUSD"].avg_sharpe.iloc[0] == 1.0


def test_main_aggregates(monkeypatch, tmp_path):
    def fake_run(cfg):
        if cfg["symbol"] == "XAUUSD":
            return {"avg_sharpe": 1.0, "worst_drawdown": -5.0}
        return {"avg_sharpe": 0.5, "worst_drawdown": -3.0}

    monkeypatch.setattr(walk_forward, "run_rolling_backtest", fake_run)
    monkeypatch.setattr(
        walk_forward, "load_config", lambda: {"symbols": ["XAUUSD", "GBPUSD"]}
    )
    log_path = tmp_path / "walk_forward_summary.csv"
    monkeypatch.setattr(walk_forward, "_LOG_PATH", log_path, raising=False)

    df = walk_forward.main()
    assert df is not None
    assert log_path.exists()
    # verify aggregation in returned DataFrame
    assert df.loc[df.symbol == "XAUUSD", "avg_sharpe"].iloc[0] == 1.0
    assert df.loc[df.symbol == "GBPUSD", "worst_drawdown"].iloc[0] == -3.0

    saved = pd.read_csv(log_path)
    assert len(saved) == 2
    assert saved.loc[saved.symbol == "XAUUSD", "avg_sharpe"].iloc[0] == 1.0


def test_main_with_app_config(monkeypatch, tmp_path):
    import pathlib

    mkdir_calls: list[Path] = []
    original_mkdir = pathlib.Path.mkdir

    def tracking_mkdir(self, *args, **kwargs):
        mkdir_calls.append(self)
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "mkdir", tracking_mkdir)

    global walk_forward
    walk_forward = importlib.reload(walk_forward)
    walk_forward.init_logging = lambda: logging.getLogger("test_walk_forward")

    assert mkdir_calls == []

    config_models_stub = sys.modules.pop("config_models", None)
    pydantic_stub = sys.modules.pop("pydantic", None)
    try:
        config_models = importlib.import_module("config_models")
    finally:
        if config_models_stub is not None:
            sys.modules["config_models"] = config_models_stub
        if pydantic_stub is not None:
            sys.modules["pydantic"] = pydantic_stub

    cfg = config_models.AppConfig(
        strategy=config_models.StrategyConfig(
            symbols=["XAUUSD", "GBPUSD"], risk_per_trade=0.01
        ),
        custom_flag="value",
        symbol="fallback",
    )

    seen_cfgs: list[dict] = []

    def fake_run(cfg_sym):
        seen_cfgs.append(cfg_sym)
        cfg_sym["training"]["model_type"] = cfg_sym["symbol"]
        return {"avg_sharpe": 1.0, "worst_drawdown": -1.0}

    monkeypatch.setattr(walk_forward, "run_rolling_backtest", fake_run)
    monkeypatch.setattr(walk_forward, "load_config", lambda: cfg)

    log_path = tmp_path / "logs" / "walk_forward_summary.csv"
    monkeypatch.setattr(walk_forward, "_LOG_PATH", log_path, raising=False)

    assert not log_path.parent.exists()
    assert mkdir_calls == []

    df = walk_forward.main()

    assert df is not None
    assert len(df) == 2
    assert log_path.exists()
    assert mkdir_calls == [log_path.parent]

    assert [cfg_dict["training"]["model_type"] for cfg_dict in seen_cfgs] == [
        "XAUUSD",
        "GBPUSD",
    ]
    assert all(cfg_dict["custom_flag"] == "value" for cfg_dict in seen_cfgs)
    assert cfg.training.model_type == "lgbm"
    saved = pd.read_csv(log_path)
    assert list(saved.symbol) == ["XAUUSD", "GBPUSD"]
