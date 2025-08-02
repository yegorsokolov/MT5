import pandas as pd
import types
import sys
import logging
from pathlib import Path
import contextlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.modules["log_utils"] = types.SimpleNamespace(
    setup_logging=lambda: logging.getLogger(),
    log_exceptions=lambda f: f,
)
sys.modules["backtest"] = types.SimpleNamespace(run_rolling_backtest=lambda cfg: {})
sys.modules["mlflow"] = types.SimpleNamespace(
    set_experiment=lambda *a, **k: None,
    set_tracking_uri=lambda *a, **k: None,
    start_run=lambda *a, **k: contextlib.nullcontext(),
    log_dict=lambda *a, **k: None,
)

import walk_forward


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
