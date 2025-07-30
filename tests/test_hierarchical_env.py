import pandas as pd
import numpy as np
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# stub heavy deps
sb3 = types.SimpleNamespace(PPO=object, SAC=object, A2C=object)
sb3.common = types.SimpleNamespace(vec_env=types.SimpleNamespace(SubprocVecEnv=object))
sys.modules.setdefault("stable_baselines3", sb3)
sys.modules.setdefault("stable_baselines3.common", sb3.common)
sys.modules.setdefault("stable_baselines3.common.vec_env", sb3.common.vec_env)
contrib = types.SimpleNamespace(TRPO=object, RecurrentPPO=object, HierarchicalPPO=object)
sys.modules.setdefault("sb3_contrib", contrib)
sys.modules.setdefault("sb3_contrib.qrdqn", types.SimpleNamespace(QRDQN=object))
sys.modules.setdefault("duckdb", types.SimpleNamespace(connect=lambda *a, **k: None))
sys.modules.setdefault("requests", types.SimpleNamespace(get=lambda *a, **k: None))
sys.modules.setdefault("prometheus_client", types.SimpleNamespace(Counter=lambda *a, **k: object(), Gauge=lambda *a, **k: object()))

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
