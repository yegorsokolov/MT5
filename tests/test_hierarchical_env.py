import pandas as pd
import numpy as np
import sys
import types
import importlib.machinery
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# stub heavy deps
sb3 = types.ModuleType("stable_baselines3")
sb3.PPO = object
sb3.SAC = object
sb3.A2C = object
sb3.__spec__ = importlib.machinery.ModuleSpec("stable_baselines3", loader=None)
sb3.common = types.ModuleType("stable_baselines3.common")
sb3.common.vec_env = types.ModuleType("stable_baselines3.common.vec_env")
sb3.common.vec_env.SubprocVecEnv = object
sb3.common.vec_env.DummyVecEnv = object
sb3.common.evaluation = types.ModuleType("stable_baselines3.common.evaluation")
sb3.common.evaluation.evaluate_policy = lambda *a, **k: None
sb3.common.__spec__ = importlib.machinery.ModuleSpec("stable_baselines3.common", loader=None)
sb3.common.vec_env.__spec__ = importlib.machinery.ModuleSpec("stable_baselines3.common.vec_env", loader=None)
sb3.common.evaluation.__spec__ = importlib.machinery.ModuleSpec("stable_baselines3.common.evaluation", loader=None)
sys.modules.setdefault("stable_baselines3", sb3)
sys.modules.setdefault("stable_baselines3.common", sb3.common)
sys.modules.setdefault("stable_baselines3.common.vec_env", sb3.common.vec_env)
sys.modules.setdefault("stable_baselines3.common.evaluation", sb3.common.evaluation)
contrib = types.ModuleType("sb3_contrib")
contrib.TRPO = object
contrib.RecurrentPPO = object
contrib.HierarchicalPPO = object
contrib.__spec__ = importlib.machinery.ModuleSpec("sb3_contrib", loader=None)
sys.modules.setdefault("sb3_contrib", contrib)
qrdqn_mod = types.ModuleType("sb3_contrib.qrdqn")
qrdqn_mod.QRDQN = object
qrdqn_mod.__spec__ = importlib.machinery.ModuleSpec("sb3_contrib.qrdqn", loader=None)
sys.modules.setdefault("sb3_contrib.qrdqn", qrdqn_mod)
duckdb_stub = types.ModuleType("duckdb")
duckdb_stub.connect = lambda *a, **k: None
duckdb_stub.__spec__ = importlib.machinery.ModuleSpec("duckdb", loader=None)
sys.modules.setdefault("duckdb", duckdb_stub)
requests_stub = types.ModuleType("requests")
requests_stub.get = lambda *a, **k: None
requests_stub.__spec__ = importlib.machinery.ModuleSpec("requests", loader=None)
sys.modules.setdefault("requests", requests_stub)
sys.modules.setdefault("prometheus_client", types.SimpleNamespace(Counter=lambda *a, **k: object(), Gauge=lambda *a, **k: object()))
torch_stub = types.ModuleType("torch")
torch_stub.manual_seed = lambda *a, **k: None
torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda *a, **k: None)
torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
sys.modules.setdefault("torch", torch_stub)
mlflow_stub = types.ModuleType("mlflow")
mlflow_stub.set_tracking_uri = lambda *a, **k: None
mlflow_stub.set_experiment = lambda *a, **k: None
mlflow_stub.start_run = lambda *a, **k: None
mlflow_stub.__spec__ = importlib.machinery.ModuleSpec("mlflow", loader=None)
sys.modules.setdefault("mlflow", mlflow_stub)
sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda *a, **k: {}))
def _dummy_validator(*args, **kwargs):
    def wrap(fn):
        return fn
    return wrap

sys.modules.setdefault(
    "pydantic",
    types.SimpleNamespace(
        BaseModel=object,
        Field=lambda *a, **k: None,
        field_validator=_dummy_validator,
        ConfigDict=dict,
        ValidationError=Exception,
    ),
)
utils_stub = types.ModuleType("utils")
utils_stub.load_config = lambda *a, **k: {}
sys.modules.setdefault("utils", utils_stub)
data_stub = types.ModuleType("data")
history_stub = types.ModuleType("data.history")
history_stub.load_history_parquet = lambda *a, **k: None
history_stub.save_history_parquet = lambda *a, **k: None
history_stub.load_history_config = lambda *a, **k: pd.DataFrame()
features_stub = types.ModuleType("data.features")
features_stub.make_features = lambda df: df
data_stub.history = history_stub
data_stub.features = features_stub
sys.modules.setdefault("data", data_stub)
sys.modules.setdefault("data.history", history_stub)
sys.modules.setdefault("data.features", features_stub)
dummy_space = type("DummySpace", (), {"__init__": lambda self, *a, **k: None})
gym_stub = types.ModuleType("gym")
gym_stub.Env = object
gym_stub.__spec__ = importlib.machinery.ModuleSpec("gym", loader=None)
spaces_mod = types.ModuleType("gym.spaces")
spaces_mod.Box = dummy_space
spaces_mod.Dict = dummy_space
spaces_mod.Discrete = dummy_space
spaces_mod.__spec__ = importlib.machinery.ModuleSpec("gym.spaces", loader=None)
gym_stub.spaces = spaces_mod
sys.modules.setdefault("gym", gym_stub)
sys.modules.setdefault("gym.spaces", spaces_mod)

from train_rl import HierarchicalTradingEnv


def test_hierarchical_step():
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020-01-01", periods=3, freq="min"),
        "Symbol": ["A"] * 3,
        "mid": [1.0, 1.1, 1.2],
        "return": [0.0, 0.1, -0.1],
    })
    env = HierarchicalTradingEnv(df, ["return"], max_position=1.0)
    env.reset()
    action = {"manager": 2, "worker": np.array([1.0], dtype=np.float32)}
    obs, reward, done, _ = env.step(action)
    assert env.positions[0] == 1.0


def test_slippage_factor_cost():
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020-01-01", periods=2, freq="min"),
        "Symbol": ["A"] * 2,
        "mid": [1.0, 1.0],
        "return": [0.0, 0.0],
    })
    np.random.seed(0)
    expected_slip = np.abs(np.random.normal(scale=0.1, size=1))[0]
    np.random.seed(0)
    env = HierarchicalTradingEnv(
        df,
        ["return"],
        max_position=1.0,
        slippage_factor=0.1,
    )
    env.reset()
    action = {"manager": 2, "worker": np.array([1.0], dtype=np.float32)}
    obs, reward, done, info = env.step(action)
    expected_cost = env.transaction_cost + expected_slip
    assert np.isclose(info["transaction_costs"][0], expected_cost)


def test_spread_execution_price():
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020-01-01", periods=2, freq="min"),
        "Symbol": ["A"] * 2,
        "mid": [1.0, 1.0],
        "spread": [0.1, 0.1],
        "return": [0.0, 0.0],
    })
    env = HierarchicalTradingEnv(
        df,
        ["return", "spread"],
        max_position=1.0,
        spread_source="column",
    )
    env.reset()
    action = {"manager": 2, "worker": np.array([1.0], dtype=np.float32)}
    obs, reward, done, info = env.step(action)
    expected_cost = env.transaction_cost + 0.05
    assert np.isclose(info["transaction_costs"][0], expected_cost)


def test_single_env_uses_dummy(monkeypatch, tmp_path):
    import importlib

    train_rl = importlib.import_module("train_rl")
    calls = {"dummy": 0, "subproc": 0}

    class DummyVecEnv:
        def __init__(self, env_fns):
            calls["dummy"] += 1
            self.env = env_fns[0]()
            self.action_space = self.env.action_space

        def reset(self):
            return self.env.reset()

        def step(self, action):
            return self.env.step(action)

    class SubprocVecEnv:
        def __init__(self, env_fns):
            calls["subproc"] += 1
            self.env = env_fns[0]()
            self.action_space = self.env.action_space

        def reset(self):
            return self.env.reset()

        def step(self, action):
            return self.env.step(action)

    monkeypatch.setattr(train_rl, "DummyVecEnv", DummyVecEnv)
    monkeypatch.setattr(train_rl, "SubprocVecEnv", SubprocVecEnv)

    class DummyAlgo:
        def __init__(self, policy, env, verbose=0, seed=0, **kwargs):
            pass

        def learn(self, total_timesteps):
            pass

        def save(self, path):
            pass

    monkeypatch.setattr(train_rl, "A2C", DummyAlgo)
    monkeypatch.setattr(train_rl, "PPO", DummyAlgo)
    monkeypatch.setattr(train_rl, "evaluate_policy", lambda *a, **k: None)

    mlflow_stub = types.SimpleNamespace(
        set_tracking_uri=lambda *a, **k: None,
        set_experiment=lambda *a, **k: None,
        start_run=lambda *a, **k: None,
        log_param=lambda *a, **k: None,
        log_artifact=lambda *a, **k: None,
        end_run=lambda *a, **k: None,
        log_metric=lambda *a, **k: None,
    )
    monkeypatch.setattr(train_rl, "mlflow", mlflow_stub)

    monkeypatch.setattr(train_rl, "__file__", tmp_path / "train_rl.py")
    monkeypatch.setattr(
        train_rl,
        "load_config",
        lambda: {
            "seed": 0,
            "symbols": ["TEST"],
            "rl_algorithm": "A3C",
            "rl_num_envs": 1,
            "rl_steps": 1,
            "rl_transaction_cost": 0.0,
            "rl_risk_penalty": 0.0,
        },
    )
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-01", periods=3, freq="H"),
            "Symbol": ["TEST"] * 3,
            "mid": 1.0,
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

    assert calls["dummy"] == 1
    assert calls["subproc"] == 0
