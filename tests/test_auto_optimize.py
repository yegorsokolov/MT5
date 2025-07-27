import pandas as pd
import yaml
import sys
import types
import contextlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.modules["mlflow"] = types.SimpleNamespace(
    set_experiment=lambda *a, **k: None,
    start_run=lambda: contextlib.nullcontext(),
    log_params=lambda *a, **k: None,
    log_metrics=lambda *a, **k: None,
)
sys.modules["lightgbm"] = types.SimpleNamespace(LGBMClassifier=object)

import auto_optimize


def test_auto_optimize_updates_config(monkeypatch, tmp_path):
    log_path = tmp_path / "hist.csv"
    monkeypatch.setattr(auto_optimize, "_LOG_PATH", log_path, raising=False)

    cfg_path = tmp_path / "cfg.yaml"
    cfg = {
        "threshold": 0.55,
        "trailing_stop_pips": 20,
        "rsi_buy": 55,
        "rl_max_position": 1.0,
        "rl_risk_penalty": 0.1,
        "rl_transaction_cost": 0.0001,
        "backtest_window_months": 6,
        "symbol": "XAUUSD",
    }
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    monkeypatch.setattr(auto_optimize, "load_config", lambda: yaml.safe_load(open(cfg_path)))

    def fake_update(key, value, reason):
        if key == "rl_max_position":
            raise ValueError("risk param")
        data = yaml.safe_load(open(cfg_path))
        data[key] = value
        with open(cfg_path, "w") as f:
            yaml.safe_dump(data, f)

    monkeypatch.setattr(auto_optimize, "update_config", fake_update)
    monkeypatch.setattr(auto_optimize, "train_model", lambda: None)

    base_metrics = {"sharpe": 0.1}
    base_returns = pd.Series([0.0] * 10)
    best_metrics = {"sharpe": 0.2}
    best_returns = pd.Series([0.2] * 10)

    def fake_backtest(cfg, return_returns=False):
        if cfg.get("threshold") == 0.55:
            if return_returns:
                return base_metrics, base_returns
            return base_metrics
        if return_returns:
            return best_metrics, best_returns
        return best_metrics

    monkeypatch.setattr(auto_optimize, "run_backtest", fake_backtest)

    def fake_cv(cfg):
        if cfg.get("threshold") == 0.55:
            return {"avg_sharpe": 0.5}
        return {"avg_sharpe": 1.5}

    monkeypatch.setattr(auto_optimize, "run_rolling_backtest", fake_cv)

    class DummyTrial:
        params = {
            "threshold": 0.6,
            "trailing_stop_pips": 15,
            "rsi_buy": 60,
            "rl_max_position": 1.5,
            "rl_risk_penalty": 0.15,
            "rl_transaction_cost": 0.0002,
            "backtest_window_months": 6,
        }

        def suggest_float(self, name, low, high):
            return self.params[name]

        def suggest_int(self, name, low, high):
            return self.params[name]

    class DummyStudy:
        def optimize(self, func, n_trials, **kwargs):
            for _ in range(n_trials):
                func(DummyTrial())

    monkeypatch.setattr(auto_optimize.optuna, "create_study", lambda **_: DummyStudy())

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(auto_optimize.mlflow, "set_experiment", lambda *_: None)
    monkeypatch.setattr(auto_optimize.mlflow, "start_run", lambda: DummyRun())
    monkeypatch.setattr(auto_optimize.mlflow, "log_params", lambda params: None)
    monkeypatch.setattr(auto_optimize.mlflow, "log_metrics", lambda metrics: None)

    auto_optimize.main()

    assert log_path.exists()
    df = pd.read_csv(log_path)
    assert "rl_max_position" in df.columns
    cfg_new = yaml.safe_load(open(cfg_path))
    assert cfg_new["rl_risk_penalty"] == 0.15
    # rl_max_position update should have been blocked
    assert cfg_new["rl_max_position"] == 1.0

