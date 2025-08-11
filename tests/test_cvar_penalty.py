import pandas as pd
import numpy as np
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# stub heavy dependencies
sb3 = types.SimpleNamespace(PPO=object, SAC=object, A2C=object)
sb3.common = types.SimpleNamespace(vec_env=types.SimpleNamespace(SubprocVecEnv=object))
sys.modules.setdefault("stable_baselines3", sb3)
sys.modules.setdefault("stable_baselines3.common", sb3.common)
sys.modules.setdefault("stable_baselines3.common.vec_env", sb3.common.vec_env)
sb3.common.evaluation = types.SimpleNamespace(evaluate_policy=lambda *a, **k: None)
sys.modules.setdefault("stable_baselines3.common.evaluation", sb3.common.evaluation)
contrib = types.SimpleNamespace(TRPO=object, RecurrentPPO=object, HierarchicalPPO=object)
sys.modules.setdefault("sb3_contrib", contrib)
sys.modules.setdefault("sb3_contrib.qrdqn", types.SimpleNamespace(QRDQN=object))
sys.modules.setdefault("duckdb", types.SimpleNamespace(connect=lambda *a, **k: None))
sys.modules.setdefault("requests", types.SimpleNamespace(get=lambda *a, **k: None))
sys.modules.setdefault("prometheus_client", types.SimpleNamespace(Counter=lambda *a, **k: object(), Gauge=lambda *a, **k: object()))
class _Space:
    def __init__(self, *a, **k):
        pass

gym_spaces = types.SimpleNamespace(Box=_Space, Dict=_Space, Discrete=_Space)
sys.modules.setdefault("gym", types.SimpleNamespace(Env=object, spaces=gym_spaces))
sys.modules.setdefault("torch", types.SimpleNamespace())
sys.modules.setdefault("mlflow", types.SimpleNamespace(log_param=lambda *a, **k: None, log_artifact=lambda *a, **k: None, end_run=lambda *a, **k: None))
sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda *a, **k: {}, dump=lambda *a, **k: ""))
sklearn_stub = types.SimpleNamespace(decomposition=types.SimpleNamespace(PCA=object))
sys.modules.setdefault("sklearn", sklearn_stub)
sys.modules.setdefault("sklearn.decomposition", sklearn_stub.decomposition)
sys.modules.setdefault("utils", types.SimpleNamespace(load_config=lambda: {}))
sys.modules.setdefault(
    "state_manager",
    types.SimpleNamespace(save_checkpoint=lambda *a, **k: None, load_latest_checkpoint=lambda *a, **k: None),
)
history_stub = types.SimpleNamespace(
    load_history_parquet=lambda *a, **k: None,
    save_history_parquet=lambda *a, **k: None,
    load_history_config=lambda *a, **k: pd.DataFrame(),
)
features_stub = types.SimpleNamespace(make_features=lambda df: df)
sys.modules.setdefault("data", types.SimpleNamespace())
sys.modules.setdefault("data.history", history_stub)
sys.modules.setdefault("data.features", features_stub)

from train_rl import TradingEnv


def make_env_with_returns(returns):
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020-01-01", periods=3, freq="min"),
        "Symbol": ["A"] * 3,
        "mid": [1.0, 1.0, 1.0],
        "return": [0.0, 0.0, 0.0],
    })
    env = TradingEnv(
        df,
        ["return"],
        transaction_cost=0.0,
        risk_penalty=0.0,
        var_window=100,
        cvar_penalty=1.0,
        cvar_window=len(returns) + 1,
    )
    env.reset()
    env.portfolio_returns = list(returns)
    env.equity = 1.0
    env.peak_equity = 1.0
    env.i = 1
    _, reward, _, _ = env.step([0.0])
    return reward


def test_cvar_penalty_reduces_reward():
    reward_low = make_env_with_returns([-0.01, -0.01, -0.01])
    reward_high = make_env_with_returns([-0.05, -0.05, -0.05])
    assert reward_high < reward_low
