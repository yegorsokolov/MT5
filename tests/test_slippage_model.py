import sys
import types
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.slippage_model import SlippageModel


def test_slippage_scales_with_liquidity():
    high_liq = [[(100.0, 1000), (100.01, 1000)]]
    low_liq = [[(100.0, 10), (100.01, 10)]]
    high_model = SlippageModel(high_liq)
    low_model = SlippageModel(low_liq)
    order_size = 20
    high_slip = high_model(order_size, "buy")
    low_slip = low_model(order_size, "buy")
    assert low_slip > high_slip


class DummyGauge:
    def __init__(self, *args, **kwargs):
        pass

    def set(self, value):
        pass

    def inc(self, value=1):
        pass


DummyCounter = DummyGauge

sys.modules.setdefault(
    "prometheus_client",
    types.SimpleNamespace(Gauge=DummyGauge, Counter=DummyCounter),
)

sys.modules.setdefault("utils", types.SimpleNamespace(load_config=lambda: {}))
sys.modules.setdefault(
    "utils.resource_monitor",
    types.SimpleNamespace(
        monitor=types.SimpleNamespace(capability_tier=lambda: "lite")
    ),
)
sys.modules.setdefault(
    "analysis.strategy_evaluator", types.SimpleNamespace(StrategyEvaluator=object)
)
sys.modules.setdefault(
    "strategy.router", types.SimpleNamespace(StrategyRouter=object, FeatureDict=dict)
)
sys.modules.setdefault(
    "state_manager",
    types.SimpleNamespace(
        load_router_state=lambda *a, **k: None, save_router_state=lambda *a, **k: None
    ),
)
sys.modules.setdefault("event_store", types.SimpleNamespace())
sys.modules.setdefault(
    "event_store.event_writer", types.SimpleNamespace(record=lambda *a, **k: None)
)
sys.modules.setdefault("model_registry", types.SimpleNamespace(ModelRegistry=object))
sys.modules.setdefault(
    "execution.execution_optimizer",
    types.SimpleNamespace(
        ExecutionOptimizer=type(
            "EO",
            (),
            {
                "get_params": lambda self: {"limit_offset": 0.0, "slice_size": None},
                "schedule_nightly": lambda self: None,
            },
        ),
        OptimizationLoopHandle=type(
            "OptHandle", (), {"stop": lambda self: None, "join": lambda self: None}
        ),
    ),
)
sys.modules.setdefault(
    "crypto_utils",
    types.SimpleNamespace(
        _load_key=lambda *a, **k: None,
        encrypt=lambda *a, **k: b"",
        decrypt=lambda *a, **k: b"",
    ),
)
sys.modules.setdefault(
    "mt5.crypto_utils",
    types.SimpleNamespace(
        _load_key=lambda *a, **k: None,
        encrypt=lambda *a, **k: b"",
        decrypt=lambda *a, **k: b"",
    ),
)

sk_module = types.ModuleType("sklearn")
sk_module.pipeline = types.SimpleNamespace(Pipeline=object)
sk_module.preprocessing = types.SimpleNamespace(StandardScaler=object)
sys.modules["sklearn"] = sk_module
sys.modules["sklearn.pipeline"] = sk_module.pipeline
sys.modules["sklearn.preprocessing"] = sk_module.preprocessing

data_mod = types.ModuleType("data")
data_history = types.ModuleType("data.history")
data_history.load_history_parquet = lambda *a, **k: None
data_history.load_history_config = lambda *a, **k: None
data_features = types.ModuleType("data.features")
data_features.make_features = lambda df: df
data_feature_scaler = types.ModuleType("data.feature_scaler")
data_feature_scaler.FeatureScaler = object
data_mod.history = data_history
data_mod.features = data_features
data_mod.feature_scaler = data_feature_scaler
sys.modules["data"] = data_mod
sys.modules["data.history"] = data_history
sys.modules["data.features"] = data_features
sys.modules["data.feature_scaler"] = data_feature_scaler
ray_stub = types.SimpleNamespace(remote=lambda f: f)
sys.modules.setdefault(
    "ray_utils",
    types.SimpleNamespace(ray=ray_stub, init=lambda **k: None, shutdown=lambda: None),
)

spec = importlib.util.spec_from_file_location(
    "backtest", Path(__file__).resolve().parents[1] / "mt5" / "backtest.py"
)
sys.modules["lightgbm"] = types.SimpleNamespace(LGBMClassifier=object)
sys.modules["log_utils"] = types.SimpleNamespace(
    setup_logging=lambda: None, log_exceptions=lambda f: f
)
backtest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backtest)
backtest.log_backtest_stats = lambda *a, **k: None


class DummyModel:
    def predict_proba(self, X):
        return np.tile([0.4, 0.6], (len(X), 1))


def _make_df() -> pd.DataFrame:
    data = {
        "Bid": [100.0, 100.02, 100.05, 99.90, 99.95, 100.00, 99.80],
        "Ask": [100.02, 100.04, 100.07, 99.92, 99.97, 100.02, 99.82],
        "BidVolume": [200] * 7,
        "AskVolume": [200] * 7,
    }
    df = pd.DataFrame(data)
    df["mid"] = (df["Bid"] + df["Ask"]) / 2
    df["return"] = df["mid"].pct_change().fillna(0)
    for ma in [5, 10, 30, 60]:
        df[f"ma_{ma}"] = df["mid"].rolling(ma, min_periods=1).mean()
    df["volatility_30"] = df["return"].rolling(30, min_periods=1).std().fillna(0)
    df["spread"] = df["Ask"] - df["Bid"]
    df["rsi_14"] = 50
    df["news_sentiment"] = 0
    return df


def test_slippage_impacts_pnl():
    model = DummyModel()
    df = _make_df()
    cfg = {"threshold": 0.5, "trailing_stop_pips": 1000, "order_size": 100}
    metrics_no, _ = backtest.backtest_on_df(df, model, cfg, return_returns=True)

    snaps_buy = [[(row.Ask, 10), (row.Ask + 1.0, 10)] for row in df.itertuples()]
    snaps_sell = [[(row.Bid, 10), (row.Bid - 1.0, 10)] for row in df.itertuples()]
    model_buy = SlippageModel(snaps_buy)
    model_sell = SlippageModel(snaps_sell)

    def slip(order_size, side):
        return (
            model_buy(order_size, side)
            if side == "buy"
            else model_sell(order_size, side)
        )

    metrics_slip, _ = backtest.backtest_on_df(
        df, model, cfg, slippage_model=slip, return_returns=True
    )
    assert metrics_slip["total_return"] < metrics_no["total_return"]
