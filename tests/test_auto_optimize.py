import importlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging
import types

import pandas as pd
import yaml

from tests.yaml_helpers import ensure_real_yaml

MLFLOW_STATE: dict[str, Any] = {}


def _reset_mlflow_state() -> None:
    MLFLOW_STATE.clear()
    MLFLOW_STATE.update(
        start_calls=0,
        end_calls=0,
        tracking_uri=None,
        experiment=None,
        experiments=[],
        active=None,
        params=[],
        metrics=[],
        dicts=[],
    )


_reset_mlflow_state()


def _mlflow_set_tracking_uri(uri: str) -> None:
    MLFLOW_STATE["tracking_uri"] = uri


def _mlflow_set_experiment(name: str) -> None:
    MLFLOW_STATE["experiment"] = name
    MLFLOW_STATE.setdefault("experiments", []).append(name)


def _mlflow_start_run():
    if MLFLOW_STATE["active"] is not None:
        raise RuntimeError("nested run")
    MLFLOW_STATE["start_calls"] += 1
    MLFLOW_STATE["active"] = object()
    return None


def _mlflow_end_run() -> None:
    if MLFLOW_STATE["active"] is not None:
        MLFLOW_STATE["end_calls"] += 1
        MLFLOW_STATE["active"] = None


def _mlflow_active_run():
    return MLFLOW_STATE["active"]


def _mlflow_log_params(params):
    MLFLOW_STATE.setdefault("params", []).append(dict(params))


def _mlflow_log_param(key, value):
    MLFLOW_STATE.setdefault("params", []).append({key: value})


def _mlflow_log_metrics(metrics):
    MLFLOW_STATE.setdefault("metrics", []).append(dict(metrics))


def _mlflow_log_metric(key, value, step=None):
    record = {key: value}
    if step is not None:
        record["step"] = step
    MLFLOW_STATE.setdefault("metrics", []).append(record)


def _mlflow_log_dict(payload, name):
    MLFLOW_STATE.setdefault("dicts", []).append((payload, name))


mlflow_stub = types.ModuleType("mlflow")
mlflow_stub.set_tracking_uri = _mlflow_set_tracking_uri
mlflow_stub.set_experiment = _mlflow_set_experiment
mlflow_stub.start_run = _mlflow_start_run
mlflow_stub.end_run = _mlflow_end_run
mlflow_stub.active_run = _mlflow_active_run
mlflow_stub.log_param = _mlflow_log_param
mlflow_stub.log_params = _mlflow_log_params
mlflow_stub.log_metric = _mlflow_log_metric
mlflow_stub.log_metrics = _mlflow_log_metrics
mlflow_stub.log_dict = _mlflow_log_dict
mlflow_stub.log_artifact = lambda *a, **k: None
mlflow_stub.log_artifacts = lambda *a, **k: None
sys.modules["mlflow"] = mlflow_stub

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
sys.modules["mt5.train"] = train_stub

utils_stub = types.ModuleType("utils")
utils_stub.load_config = lambda: {}
utils_stub.update_config = lambda *a, **k: None
utils_stub.PROJECT_ROOT = Path.cwd()
utils_stub.sanitize_config = lambda cfg, **_: cfg
sys.modules["utils"] = utils_stub

backtest_stub = types.ModuleType("backtest")
backtest_stub.run_backtest = lambda *a, **k: {}
backtest_stub.run_rolling_backtest = lambda *a, **k: {}
sys.modules["backtest"] = backtest_stub
sys.modules["mt5.backtest"] = backtest_stub

log_utils_stub = types.ModuleType("log_utils")
log_utils_stub.setup_logging = lambda *a, **k: None
log_utils_stub.log_exceptions = lambda func: func
sys.modules["log_utils"] = log_utils_stub
sys.modules["mt5.log_utils"] = log_utils_stub

sys.modules["lightgbm"] = types.SimpleNamespace(LGBMClassifier=object)

import analytics.mlflow_client as mlflow_client

importlib.reload(mlflow_client)

from mt5 import auto_optimize


def test_auto_optimize_updates_config(monkeypatch, tmp_path):
    ensure_real_yaml()
    _reset_mlflow_state()

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


def test_auto_optimize_uses_single_mlflow_run(monkeypatch, tmp_path):
    ensure_real_yaml()
    _reset_mlflow_state()

    monkeypatch.setattr(
        auto_optimize,
        "init_logging",
        lambda: logging.getLogger("test_auto_optimize_single_run"),
    )

    log_path = tmp_path / "hist.csv"
    monkeypatch.setattr(auto_optimize, "_LOG_PATH", log_path, raising=False)

    base_cfg = {
        "threshold": 0.55,
        "trailing_stop_pips": 20,
        "rsi_buy": 55,
        "rl_max_position": 1.0,
        "rl_risk_penalty": 0.1,
        "rl_transaction_cost": 0.0001,
        "rl_max_kl": 0.01,
        "backtest_window_months": 6,
    }

    def load_config_stub():
        return dict(base_cfg)

    monkeypatch.setattr(auto_optimize, "load_config", load_config_stub)
    monkeypatch.setattr(auto_optimize, "update_config", lambda *a, **k: None)

    def training_stub():
        started = auto_optimize.mlflow_client.start_run("training", {"stage": "train"})
        if started:
            auto_optimize.mlflow_client.end_run()

    monkeypatch.setattr(auto_optimize, "train_model", training_stub)

    base_metrics = {"sharpe": 0.1}
    base_returns = pd.Series([0.0, 0.0])
    best_metrics = {"sharpe": 0.2}
    best_returns = pd.Series([0.2, 0.2])

    def fake_backtest(cfg, return_returns=False):
        if cfg.get("threshold") == base_cfg["threshold"]:
            if return_returns:
                return base_metrics, base_returns
            return base_metrics
        if return_returns:
            return best_metrics, best_returns
        return best_metrics

    monkeypatch.setattr(auto_optimize, "run_backtest", fake_backtest)

    def fake_cv(cfg):
        if cfg.get("threshold") == base_cfg["threshold"]:
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

    auto_optimize.main()

    assert log_path.exists()
    assert MLFLOW_STATE["start_calls"] == 1
    assert MLFLOW_STATE["end_calls"] == 1


def test_auto_optimize_warns_when_mlflow_missing(monkeypatch, tmp_path, caplog):
    ensure_real_yaml()
    _reset_mlflow_state()

    mlflow_module = sys.modules.pop("mlflow", None)
    analytics_pkg = sys.modules.get("analytics")
    analytics_attr_present = False
    analytics_mlflow_attr = None
    if analytics_pkg is not None and hasattr(analytics_pkg, "mlflow_client"):
        analytics_attr_present = True
        analytics_mlflow_attr = getattr(analytics_pkg, "mlflow_client")
        delattr(analytics_pkg, "mlflow_client")
    mlflow_client_module = sys.modules.pop("analytics.mlflow_client", None)

    try:
        module = importlib.reload(auto_optimize)
        assert not getattr(module, "MLFLOW_AVAILABLE", True)

        monkeypatch.setattr(
            module,
            "init_logging",
            lambda: logging.getLogger("test_auto_optimize_missing_mlflow"),
        )

        log_path = tmp_path / "hist.csv"
        monkeypatch.setattr(module, "_LOG_PATH", log_path, raising=False)

        base_cfg = {
            "threshold": 0.55,
            "trailing_stop_pips": 20,
            "rsi_buy": 55,
            "rl_max_position": 1.0,
            "rl_risk_penalty": 0.1,
            "rl_transaction_cost": 0.0001,
            "rl_max_kl": 0.01,
            "backtest_window_months": 6,
        }

        monkeypatch.setattr(module, "load_config", lambda: dict(base_cfg))
        monkeypatch.setattr(module, "train_model", lambda: None)
        monkeypatch.setattr(module, "update_config", lambda *a, **k: None)

        base_metrics = {"sharpe": 0.1}
        base_returns = pd.Series([0.0, 0.0])
        best_metrics = {"sharpe": 0.2}
        best_returns = pd.Series([0.2, 0.2])

        def fake_backtest(cfg, return_returns=False):
            if cfg.get("threshold") == base_cfg["threshold"]:
                if return_returns:
                    return base_metrics, base_returns
                return base_metrics
            if return_returns:
                return best_metrics, best_returns
            return best_metrics

        monkeypatch.setattr(module, "run_backtest", fake_backtest)

        def fake_cv(cfg):
            if cfg.get("threshold") == base_cfg["threshold"]:
                return {"avg_sharpe": 0.5}
            return {"avg_sharpe": 1.5}

        monkeypatch.setattr(module, "run_rolling_backtest", fake_cv)
        monkeypatch.setattr(
            module,
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

        monkeypatch.setattr(module.optuna, "create_study", lambda **_: DummyStudy())

        with caplog.at_level(logging.WARNING):
            module.main()

        assert log_path.exists()
        assert any(
            "MLflow is not installed" in record.getMessage()
            for record in caplog.records
        )
    finally:
        sys.modules["mlflow"] = mlflow_module or mlflow_stub
        if mlflow_client_module is not None:
            sys.modules["analytics.mlflow_client"] = mlflow_client_module
        else:
            sys.modules.setdefault("analytics.mlflow_client", mlflow_client)
        if analytics_pkg is not None:
            if analytics_attr_present:
                setattr(analytics_pkg, "mlflow_client", analytics_mlflow_attr)
            elif not hasattr(analytics_pkg, "mlflow_client"):
                setattr(analytics_pkg, "mlflow_client", mlflow_client)
        importlib.reload(auto_optimize)

