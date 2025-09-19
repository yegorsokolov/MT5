import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import contextlib
import logging
import types

import pandas as pd
import yaml

from tests.yaml_helpers import ensure_real_yaml

scipy_stats_stub = types.ModuleType("scipy.stats")


def _dummy_ttest_ind(*_a, **_k):
    return types.SimpleNamespace(pvalue=0.01)


scipy_stats_stub.ttest_ind = _dummy_ttest_ind  # type: ignore[attr-defined]
scipy_stub = types.ModuleType("scipy")
scipy_stub.stats = scipy_stats_stub  # type: ignore[attr-defined]
sys.modules["scipy"] = scipy_stub
sys.modules["scipy.stats"] = scipy_stats_stub

optuna_stub = types.ModuleType("optuna")
optuna_stub.trial = types.SimpleNamespace(Trial=object)
optuna_stub.create_study = lambda *a, **k: None
sys.modules["optuna"] = optuna_stub

train_stub = types.ModuleType("train")
train_stub.main = lambda *a, **k: None
sys.modules["train"] = train_stub

utils_stub = types.ModuleType("utils")
utils_stub.load_config = lambda: {}
utils_stub.update_config = lambda *a, **k: None
sys.modules["utils"] = utils_stub

backtest_stub = types.ModuleType("backtest")
backtest_stub.run_backtest = lambda *a, **k: {}
backtest_stub.run_rolling_backtest = lambda *a, **k: {}
sys.modules["backtest"] = backtest_stub

log_utils_stub = types.ModuleType("log_utils")
log_utils_stub.setup_logging = lambda *a, **k: None
log_utils_stub.log_exceptions = lambda func: func
sys.modules["log_utils"] = log_utils_stub

sys.modules["mlflow"] = types.SimpleNamespace(
    set_experiment=lambda *a, **k: None,
    start_run=lambda: contextlib.nullcontext(),
    log_params=lambda *a, **k: None,
    log_metrics=lambda *a, **k: None,
)
sys.modules["lightgbm"] = types.SimpleNamespace(LGBMClassifier=object)

import auto_optimize


def test_auto_optimize_updates_config(monkeypatch, tmp_path):
    ensure_real_yaml()

    monkeypatch.setattr(
        auto_optimize,
        "init_logging",
        lambda: logging.getLogger("test_auto_optimize"),
    )

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
        "rl_max_kl": 0.01,
        "backtest_window_months": 6,
        "symbol": "XAUUSD",
    }
    dumped_text = yaml.safe_dump(cfg)
    assert dumped_text
    cfg_path.write_text(dumped_text)
    loaded_cfg = yaml.safe_load(cfg_path.read_text())
    assert loaded_cfg and loaded_cfg.get("threshold") == 0.55

    captured_dump: dict[str, dict] = {}

    class DummyConfig:
        def __init__(self, data):
            self._data = data

        def model_dump(self):
            dumped = dict(self._data)
            captured_dump.setdefault("data", dumped)
            return dumped

    def load_config_stub():
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return DummyConfig(data)

    monkeypatch.setattr(auto_optimize, "load_config", load_config_stub)

    updates: list[dict] = []

    def fake_update(key, value, reason):
        if key == "rl_max_position":
            raise ValueError("risk param")
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        data[key] = value
        updates.append(dict(data))
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

    monkeypatch.setattr(auto_optimize, "update_config", fake_update)
    monkeypatch.setattr(auto_optimize, "train_model", lambda: None)

    base_metrics = {"sharpe": 0.1}
    base_returns = pd.Series([0.0] * 10)
    best_metrics = {"sharpe": 0.2}
    best_returns = pd.Series([0.2] * 10)
    seen_cfgs: list[dict] = []

    def fake_backtest(cfg, return_returns=False):
        seen_cfgs.append(dict(cfg))
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
    monkeypatch.setattr(
        auto_optimize,
        "ttest_ind",
        lambda *a, **k: types.SimpleNamespace(pvalue=0.01),
    )

    class DummyTrial:
        params = {
            "threshold": 0.6,
            "trailing_stop_pips": 15,
            "rsi_buy": 60,
            "rl_max_position": 1.5,
            "rl_risk_penalty": 0.15,
            "rl_transaction_cost": 0.0002,
            "rl_max_kl": 0.02,
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
    assert "data" in captured_dump
    assert "threshold" in captured_dump["data"]
    assert seen_cfgs, "run_backtest should have been called"
    assert seen_cfgs[0]["threshold"] == 0.55
    assert seen_cfgs[-1]["threshold"] == 0.6
    assert updates, "update_config should have been invoked"
    assert cfg_new["rl_risk_penalty"] == 0.15
    # rl_max_position update should have been blocked
    assert cfg_new["rl_max_position"] == 1.0

