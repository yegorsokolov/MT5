import pandas as pd
import numpy as np
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
sys.modules.setdefault(
    "prometheus_client",
    types.SimpleNamespace(
        Counter=lambda *a, **k: types.SimpleNamespace(inc=lambda *a, **k: None),
        Gauge=lambda *a, **k: types.SimpleNamespace(set=lambda *a, **k: None),
    ),
)
class _Space:
    def __init__(self, *a, **k):
        pass

gym_spaces = types.SimpleNamespace(Box=_Space, Dict=_Space, Discrete=_Space)
sys.modules.setdefault("gym", types.SimpleNamespace(Env=object, spaces=gym_spaces, Wrapper=object))
sys.modules.setdefault("torch", types.SimpleNamespace())
sys.modules.setdefault("mlflow", types.SimpleNamespace(log_param=lambda *a, **k: None, log_artifact=lambda *a, **k: None, end_run=lambda *a, **k: None))
monitor_stub = types.SimpleNamespace(
    start=lambda: None,
    capability_tier="lite",
    capabilities=types.SimpleNamespace(capability_tier=lambda: "lite", ddp=lambda: False),
)
utils_stub = types.SimpleNamespace(load_config=lambda: {}, resource_monitor=types.SimpleNamespace(monitor=monitor_stub))
sys.modules.setdefault("utils", utils_stub)
sys.modules.setdefault("utils.resource_monitor", types.SimpleNamespace(monitor=monitor_stub))
sys.modules.setdefault("state_manager", types.SimpleNamespace(save_checkpoint=lambda *a, **k: None, load_latest_checkpoint=lambda *a, **k: None))
sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda *a, **k: {}, dump=lambda *a, **k: ""))
sys.modules.setdefault("requests", types.SimpleNamespace())
sys.modules.setdefault("crypto_utils", types.SimpleNamespace(_load_key=lambda *a, **k: b"", encrypt=lambda *a, **k: b"", decrypt=lambda *a, **k: b""))
plugins_pkg = types.ModuleType("plugins")
plugins_rl_risk = types.ModuleType("plugins.rl_risk")
plugins_rl_risk.RiskEnv = object
plugins_pkg.rl_risk = plugins_rl_risk
sys.modules.setdefault("plugins", plugins_pkg)
sys.modules.setdefault("plugins.rl_risk", plugins_rl_risk)
metrics_stub = types.SimpleNamespace(record_metric=lambda *a, **k: None, TS_PATH="")
sys.modules.setdefault("analytics", types.SimpleNamespace(metrics_store=metrics_stub))
sys.modules.setdefault("analytics.metrics_store", metrics_stub)
history_stub = types.SimpleNamespace(load_history_parquet=lambda *a, **k: None, save_history_parquet=lambda *a, **k: None, load_history_config=lambda *a, **k: pd.DataFrame())
features_stub = types.SimpleNamespace(make_features=lambda df: df)
sys.modules.setdefault("data", types.SimpleNamespace())
sys.modules.setdefault("data.history", history_stub)
sys.modules.setdefault("data.features", features_stub)

from train_rl import TradingEnv


def test_close_action_realizes_return():
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020-01-01", periods=3, freq="H"),
        "Symbol": ["A"] * 3,
        "mid": [100.0, 101.0, 101.0],
        "return": [0.0, 0.0, 0.0],
    })
    env = TradingEnv(df, ["return"], transaction_cost=0.0, risk_penalty=0.0, exit_penalty=0.0)
    env.reset()
    env.step([1.0])
    _, reward, _, info = env.step([0.0, 1.0])
    assert info["portfolio_return"] > 0
    assert env.positions[0] == 0.0
