import sys
import types
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

metrics_stub = types.ModuleType("analytics.metrics_store")
metrics_stub.record_metric = lambda *a, **k: None
metrics_stub.TS_PATH = ""
analytics_pkg = types.ModuleType("analytics")
analytics_pkg.metrics_store = metrics_stub
analytics_pkg.__path__ = []
sys.modules["analytics.metrics_store"] = metrics_stub
sys.modules["analytics"] = analytics_pkg

class _Space:
    def __init__(self, *a, **k):
        pass

spaces = types.SimpleNamespace(Box=_Space, Dict=_Space, MultiBinary=_Space, Discrete=_Space)
sys.modules.setdefault("gym", types.SimpleNamespace(Env=object, spaces=spaces))
from rl.trading_env import TradingEnv
from rl.risk_shaped_env import RiskShapedTradingEnv


def run_episode(env, actions):
    obs = env.reset()
    total = 0.0
    info = {}
    for act in actions:
        obs, reward, done, info = env.step(act)
        total += reward
        if done:
            break
    return total, info


def test_risk_shaped_env_penalizes_risk():
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020-01-01", periods=4, freq="H"),
        "Symbol": ["A"] * 4,
        "mid": [100.0, 90.0, 90.0, 90.0],
        "return": [0.0, -0.1, 0.0, 0.0],
    })
    base_env = TradingEnv(df, ["return"], transaction_cost=0.0, risk_penalty=0.0, exit_penalty=0.0)
    risk_env = RiskShapedTradingEnv(
        df,
        ["return"],
        transaction_cost=0.0,
        risk_penalty=0.0,
        exit_penalty=0.0,
        drawdown_penalty=1.0,
        vol_penalty=1.0,
        slippage_penalty=0.1,
        vol_window=2,
    )
    actions = [[1.0], [1.0], [0.0, 1.0]]
    base_total, _ = run_episode(base_env, actions)
    risk_total, info = run_episode(risk_env, actions)
    assert risk_total < base_total
    assert info["drawdown_cost"] > 0
    assert info["volatility_cost"] > 0
    assert info["slippage_cost"] > 0
