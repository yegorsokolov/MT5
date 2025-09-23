import types
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging


class DummyModel:
    def predict(self, obs, deterministic=True):
        return 0, None

    @classmethod
    def load(cls, path, env=None):
        return cls()


class StubEnv:
    def __init__(self, *args, **kwargs):
        self.equity = 1.0
        self.objectives = kwargs.get("objectives", ["return"])
        self.action_space = types.SimpleNamespace(shape=(1,))

    def reset(self):
        return np.zeros(1)

    def step(self, action):
        info = {"objectives": {"return": 1.0, "risk": -0.5, "cost": -0.2}}
        return np.zeros(1), 0.3, True, info


def test_eval_reports_tradeoffs(monkeypatch, caplog):
    sb3_stub = types.SimpleNamespace(
        PPO=types.SimpleNamespace(load=DummyModel.load),
        SAC=types.SimpleNamespace(load=DummyModel.load),
        A2C=types.SimpleNamespace(load=DummyModel.load),
    )
    sb3_contrib_stub = types.SimpleNamespace(qrdqn=types.SimpleNamespace(QRDQN=DummyModel))
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3_stub)
    monkeypatch.setitem(sys.modules, "sb3_contrib", sb3_contrib_stub)
    monkeypatch.setitem(sys.modules, "sb3_contrib.qrdqn", sb3_contrib_stub.qrdqn)

    train_stub = types.ModuleType("train_rl")
    train_stub.TradingEnv = StubEnv
    train_stub.DiscreteTradingEnv = StubEnv
    monkeypatch.setitem(sys.modules, "train_rl", train_stub)

    utils_stub = types.ModuleType("utils")
    utils_stub.load_config = lambda: {"rl_algorithm": "PPO", "rl_objectives": ["return", "risk", "cost"]}
    monkeypatch.setitem(sys.modules, "utils", utils_stub)

    history_stub = types.SimpleNamespace(
        load_history_parquet=lambda *a, **k: None,
        save_history_parquet=lambda *a, **k: None,
        load_history_config=lambda *a, **k: pd.DataFrame(),
    )
    features_stub = types.SimpleNamespace(make_features=lambda df: df)
    data_pkg = types.ModuleType("data")
    monkeypatch.setitem(sys.modules, "data", data_pkg)
    monkeypatch.setitem(sys.modules, "data.history", history_stub)
    monkeypatch.setitem(sys.modules, "data.features", features_stub)

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace())
    log_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    test_logger = logging.getLogger("test_eval_rl")
    log_utils_stub = types.ModuleType("log_utils")

    def _setup_logging(*args, **kwargs):
        log_calls.append((args, kwargs))
        return test_logger

    log_utils_stub.setup_logging = _setup_logging
    log_utils_stub.log_exceptions = lambda f: f
    metrics_stub = types.ModuleType("metrics")
    metrics_stub.ERROR_COUNT = metrics_stub.TRADE_COUNT = object()
    monkeypatch.setitem(sys.modules, "log_utils", log_utils_stub)
    monkeypatch.setitem(sys.modules, "metrics", metrics_stub)

    from mt5 import eval_rl
    from mt5.train_rl import artifact_dir

    assert not log_calls

    cfg = utils_stub.load_config()
    artifact_root = artifact_dir(cfg)
    model_path = artifact_root / "models" / "model_rl.zip"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.touch()

    with caplog.at_level("INFO"):
        eval_rl.main()
    assert len(log_calls) == 1
    assert "return_total" in caplog.text
    assert "risk_total" in caplog.text
    assert "cost_total" in caplog.text
