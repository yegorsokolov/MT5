import types
import sys

import importlib.util
from pathlib import Path
import pandas as pd
import sys
import numpy as np


def test_rl_evaluation_metrics(monkeypatch, tmp_path):
    metrics_logged = {}
    mlflow_stub = types.SimpleNamespace(
        set_tracking_uri=lambda *a, **k: None,
        set_experiment=lambda *a, **k: None,
        start_run=lambda *a, **k: None,
        log_param=lambda *a, **k: None,
        log_artifact=lambda *a, **k: None,
        end_run=lambda *a, **k: None,
        log_metric=lambda k, v: metrics_logged.setdefault(k, v),
        __spec__=types.SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_stub)
    prom_stub = types.SimpleNamespace(
        Counter=lambda *a, **k: types.SimpleNamespace(),
        Gauge=lambda *a, **k: types.SimpleNamespace(),
        __spec__=types.SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "prometheus_client", prom_stub)
    torch_stub = types.SimpleNamespace(
        manual_seed=lambda *a, **k: None,
        cuda=types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda *a, **k: None),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    class DummyBox:
        def __init__(self, low, high, shape, dtype=None):
            self.shape = shape

    gym_stub = types.SimpleNamespace(Env=object, spaces=types.SimpleNamespace(Box=DummyBox))
    monkeypatch.setitem(sys.modules, "gym", gym_stub)
    monkeypatch.setitem(sys.modules, "utils", types.SimpleNamespace(load_config=lambda: {}))
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace())
    history_stub = types.SimpleNamespace(
        load_history_parquet=lambda *a, **k: None,
        save_history_parquet=lambda *a, **k: None,
        load_history_config=lambda *a, **k: pd.DataFrame(),
    )
    features_stub = types.SimpleNamespace(make_features=lambda df: df)
    monkeypatch.setitem(sys.modules, "data", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "data.history", history_stub)
    monkeypatch.setitem(sys.modules, "data.features", features_stub)

    class DummyAlgo:
        def __init__(self, policy, env, verbose=0, seed=0, **kwargs):
            self.env = env

        def learn(self, total_timesteps):
            pass

        def predict(self, obs, deterministic=True):
            return np.zeros(self.env.action_space.shape), None

        def save(self, path):
            pass

    def eval_policy(model, env, n_eval_episodes=1, deterministic=True):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, done, _ = env.step(action)

    sb3_common = types.SimpleNamespace(
        vec_env=types.SimpleNamespace(SubprocVecEnv=lambda *a, **k: None),
        evaluation=types.SimpleNamespace(evaluate_policy=eval_policy),
    )
    sb3_stub = types.SimpleNamespace(PPO=DummyAlgo, SAC=DummyAlgo, A2C=DummyAlgo, common=sb3_common)
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3_stub)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", sb3_common)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.vec_env", sb3_common.vec_env)
    monkeypatch.setitem(
        sys.modules, "stable_baselines3.common.evaluation", sb3_common.evaluation
    )
    sb3_contrib_stub = types.SimpleNamespace(
        TRPO=DummyAlgo,
        RecurrentPPO=DummyAlgo,
        qrdqn=types.SimpleNamespace(QRDQN=DummyAlgo),
    )
    monkeypatch.setitem(sys.modules, "sb3_contrib", sb3_contrib_stub)
    monkeypatch.setitem(sys.modules, "sb3_contrib.qrdqn", sb3_contrib_stub.qrdqn)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    spec = importlib.util.spec_from_file_location(
        "train_rl", Path(__file__).resolve().parents[1] / "train_rl.py"
    )
    train_rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_rl)

    monkeypatch.setattr(train_rl, "__file__", tmp_path / "train_rl.py")
    cfg = {
        "seed": 123,
        "symbols": ["TEST"],
        "rl_algorithm": "PPO",
        "rl_steps": 10,
        "rl_transaction_cost": 0.0,
        "rl_risk_penalty": 0.0,
    }
    monkeypatch.setattr(train_rl, "load_config", lambda: cfg)

    n = 20
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-01", periods=n, freq="H"),
            "Symbol": ["TEST"] * n,
            "mid": 100.0,
            "return": 0.0,
            "ma_5": 0.0,
            "ma_10": 0.0,
            "ma_30": 0.0,
            "ma_60": 0.0,
            "volatility_30": 0.0,
            "spread": 0.0,
            "rsi_14": 0.0,
            "news_sentiment": 0.0,
            "market_regime": 0.0,
        }
    )
    monkeypatch.setattr(train_rl, "load_history_config", lambda *a, **k: df)
    monkeypatch.setattr(train_rl, "make_features", lambda d: d)

    train_rl.main()

    assert metrics_logged["cumulative_return"] == 0.0
    assert metrics_logged["sharpe_ratio"] == 0.0
    assert metrics_logged["max_drawdown"] == 0.0
