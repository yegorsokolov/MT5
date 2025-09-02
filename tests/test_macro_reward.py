import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

class _Space:
    def __init__(self, *a, **k):
        pass

spaces = types.SimpleNamespace(Box=_Space, Dict=_Space, MultiBinary=_Space, Discrete=_Space)
class _Wrapper:
    def __init__(self, env):
        self.env = env

    def reset(self, *a, **k):
        return self.env.reset(*a, **k)

    def step(self, action):
        return self.env.step(action)

sys.modules.setdefault("gym", types.SimpleNamespace(Env=object, spaces=spaces, Wrapper=_Wrapper))

metrics_stub = types.ModuleType("analytics.metrics_store")
metrics_stub.record_metric = lambda *a, **k: None
metrics_stub.TS_PATH = ""
analytics_pkg = types.ModuleType("analytics")
analytics_pkg.metrics_store = metrics_stub
analytics_pkg.__path__ = []
sys.modules["analytics.metrics_store"] = metrics_stub
sys.modules.setdefault("analytics", analytics_pkg)

from rl.trading_env import TradingEnv
from rl.macro_reward_wrapper import MacroRewardWrapper


def _make_df():
    return pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020", periods=3, freq="D"),
            "Symbol": ["A"] * 3,
            "mid": [1.0, 1.1, 1.2],
            "feat": [0.0, 0.0, 0.0],
            "macro": [0.5, 0.6, 0.7],
        }
    )


def _action(pos: float):
    return {"size": np.array([pos], dtype=np.float32), "close": np.array([0.0], dtype=np.float32)}


def test_macro_reward_alignment():
    df = _make_df()
    base_env = TradingEnv(df, features=["feat"], macro_features=["macro"])
    macro_env = MacroRewardWrapper(TradingEnv(df, features=["feat"], macro_features=["macro"]), bonus_coeff=1.0)
    base_env.reset()
    macro_env.reset()
    _, r_base, _, _ = base_env.step(_action(0.5))
    _, r_macro, _, info = macro_env.step(_action(0.5))
    assert r_macro > r_base
    assert info["macro_bonus"] > 0


def test_macro_reward_penalty_on_misalignment():
    df = _make_df()
    base_env = TradingEnv(df, features=["feat"], macro_features=["macro"])
    macro_env = MacroRewardWrapper(TradingEnv(df, features=["feat"], macro_features=["macro"]), bonus_coeff=1.0)
    base_env.reset()
    macro_env.reset()
    _, r_base, _, _ = base_env.step(_action(-0.5))
    _, r_macro, _, info = macro_env.step(_action(-0.5))
    assert r_macro < r_base
    assert info["macro_bonus"] < 0
