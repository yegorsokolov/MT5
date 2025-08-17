import pandas as pd
import numpy as np
from pathlib import Path
import sys
import types
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Create lightweight 'data' package to avoid executing heavy __init__
data_pkg = types.ModuleType("data")
data_pkg.__path__ = [str(ROOT / "data")]
sys.modules.setdefault("data", data_pkg)

spec = importlib.util.spec_from_file_location("data.order_book", ROOT / "data" / "order_book.py")
order_book = importlib.util.module_from_spec(spec)
spec.loader.exec_module(order_book)
compute_order_book_features = order_book.compute_order_book_features

import types
sys.modules.setdefault("MetaTrader5", types.SimpleNamespace())
conn_mgr_stub = types.SimpleNamespace(init=lambda *a, **k: None,
                                      get_active_broker=lambda: None,
                                      failover=lambda: None)
sys.modules.setdefault("brokers", types.SimpleNamespace(connection_manager=conn_mgr_stub))
mlflow_stub = types.ModuleType("mlflow")
mlflow_stub.__spec__ = importlib.machinery.ModuleSpec("mlflow", loader=None)
sys.modules.setdefault("mlflow", mlflow_stub)
utils_pkg = types.ModuleType("utils")
utils_pkg.__path__ = []
utils_pkg.load_config = lambda *a, **k: {}
data_backend_stub = types.SimpleNamespace(get_dataframe_module=lambda: __import__("pandas"))
utils_pkg.data_backend = data_backend_stub
sys.modules.setdefault("utils", utils_pkg)
sys.modules.setdefault("utils.data_backend", data_backend_stub)
features_stub = types.ModuleType("data.features")
features_stub.make_features = lambda df: df
sys.modules.setdefault("data.features", features_stub)
signal_queue_stub = types.ModuleType("signal_queue")
signal_queue_stub.get_signal_backend = lambda cfg: None
sys.modules.setdefault("signal_queue", signal_queue_stub)
telemetry_stub = types.ModuleType("telemetry")
telemetry_stub.get_tracer = lambda name: types.SimpleNamespace(start_as_current_span=lambda *a, **k: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self,*args: None))
telemetry_stub.get_meter = lambda name: types.SimpleNamespace(create_counter=lambda *a, **k: types.SimpleNamespace(add=lambda *a, **k: None), create_histogram=lambda *a, **k: types.SimpleNamespace(record=lambda *a, **k: None))
sys.modules.setdefault("telemetry", telemetry_stub)
torch_stub = types.ModuleType("torch")
torch_stub.manual_seed = lambda *a, **k: None
torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda *a, **k: None)
sys.modules.setdefault("torch", torch_stub)
gym_stub = types.ModuleType("gym")
gym_stub.Env = object
class _Box:
    def __init__(self, *a, **k):
        pass

class _Discrete:
    def __init__(self, *a, **k):
        pass

class _Dict:
    def __init__(self, *a, **k):
        pass

gym_stub.spaces = types.SimpleNamespace(Box=_Box, Discrete=_Discrete, Dict=_Dict)
gym_stub.Wrapper = object
sys.modules.setdefault("gym", gym_stub)
sb3_stub = types.ModuleType("stable_baselines3")
sb3_stub.PPO = sb3_stub.SAC = sb3_stub.A2C = object
sys.modules.setdefault("stable_baselines3", sb3_stub)
sys.modules.setdefault(
    "stable_baselines3.common", types.SimpleNamespace(vec_env=types.SimpleNamespace(SubprocVecEnv=object, DummyVecEnv=object), evaluation=types.SimpleNamespace(evaluate_policy=lambda *a, **k: (0, 0)))
)
sys.modules.setdefault(
    "stable_baselines3.common.vec_env",
    types.SimpleNamespace(SubprocVecEnv=object, DummyVecEnv=object),
)
sys.modules.setdefault(
    "stable_baselines3.common.evaluation",
    types.SimpleNamespace(evaluate_policy=lambda *a, **k: (0, 0)),
)
sb3_contrib_stub = types.ModuleType("sb3_contrib")
sb3_contrib_stub.qrdqn = types.SimpleNamespace(QRDQN=object)
sb3_contrib_stub.TRPO = sb3_contrib_stub.RecurrentPPO = object
sb3_contrib_stub.__path__ = []
sys.modules.setdefault("sb3_contrib", sb3_contrib_stub)
sys.modules.setdefault("sb3_contrib.qrdqn", types.SimpleNamespace(QRDQN=object))
plugins_pkg = types.ModuleType("plugins")
plugins_pkg.__path__ = []
rl_risk_stub = types.ModuleType("plugins.rl_risk")
rl_risk_stub.RiskEnv = object
sys.modules.setdefault("plugins", plugins_pkg)
sys.modules.setdefault("plugins.rl_risk", rl_risk_stub)

from realtime_train import apply_liquidity_adjustment
from train_rl import TradingEnv
from analytics import metrics_store


def test_order_book_feature_computation():
    book = pd.DataFrame(
        {
            "Timestamp": [pd.Timestamp("2024-01-01 00:00:00")],
            "BidPrice1": [100.0],
            "BidVolume1": [10.0],
            "BidPrice2": [99.5],
            "BidVolume2": [5.0],
            "AskPrice1": [100.5],
            "AskVolume1": [8.0],
            "AskPrice2": [101.0],
            "AskVolume2": [4.0],
        }
    )
    feats = compute_order_book_features(book)
    expected_imbalance = (15.0 - 12.0) / (15.0 + 12.0)
    assert np.isclose(feats.loc[0, "depth_imbalance"], expected_imbalance)
    spread1 = 100.5 - 100.0
    spread2 = 101.0 - 99.5
    total_vol = (10 + 8) + (5 + 4)
    expected_vw = (spread1 * (10 + 8) + spread2 * (5 + 4)) / total_vol
    assert np.isclose(feats.loc[0, "vw_spread"], expected_vw)


def test_liquidity_adjustments_and_metrics(tmp_path):
    df = pd.DataFrame(
        {
            "mid": [100.0, 100.0],
            "vw_spread": [0.2, 0.2],
            "market_impact": [0.05, 0.05],
            "depth_imbalance": [0.1, -0.1],
        }
    )
    calls: list[tuple[str, float]] = []
    def fake_metric(name, value, **k):
        calls.append((name, value))
    metrics_store.record_metric = fake_metric
    import realtime_train as rt
    rt.record_metric = fake_metric
    out = apply_liquidity_adjustment(df)
    assert np.isclose(out["buy_fill"].iloc[0], 100.0 + 0.1 + 0.05)
    assert {name for name, _ in calls} == {"slippage", "liquidity_usage"}


def test_trading_env_uses_liquidity_metrics(tmp_path):
    metrics_store.TS_PATH = tmp_path / "env_metrics.parquet"
    import train_rl
    train_rl.TS_PATH = metrics_store.TS_PATH
    metrics_store.record_metric = lambda *a, **k: None
    train_rl.record_metric = metrics_store.record_metric
    df = pd.DataFrame(
        {
            "Timestamp": [pd.Timestamp("2024-01-01 00:00:00"), pd.Timestamp("2024-01-01 00:01:00")],
            "Symbol": ["TEST", "TEST"],
            "mid": [100.0, 101.0],
            "return": [0.0, 0.0],
            "vw_spread": [0.2, 0.2],
            "market_impact": [0.05, 0.05],
            "depth_imbalance": [0.0, 0.0],
        }
    )
    env = TradingEnv(df, ["return", "depth_imbalance", "vw_spread", "market_impact"], spread_source="column")
    env.reset()
    _, _, _, info = env.step([1.0])
    assert np.isclose(info["execution_prices"][0], 100.0 + 0.1 + 0.05)
