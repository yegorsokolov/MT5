import sys
import types
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


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
data_mod.history = data_history
data_mod.features = data_features
sys.modules["data"] = data_mod
sys.modules["data.history"] = data_history
sys.modules["data.features"] = data_features
ray_stub = types.SimpleNamespace(remote=lambda f: f)
sys.modules.setdefault(
    "ray_utils",
    types.SimpleNamespace(ray=ray_stub, init=lambda **k: None, shutdown=lambda: None),
)

# load backtest module with dummy dependencies
spec = importlib.util.spec_from_file_location(
    "backtest", Path(__file__).resolve().parents[1] / "backtest.py"
)
sys.modules["lightgbm"] = types.SimpleNamespace(LGBMClassifier=object)
sys.modules["log_utils"] = types.SimpleNamespace(
    setup_logging=lambda: None, log_exceptions=lambda f: f
)
backtest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backtest)


class DummyModel:
    def predict_proba(self, X):  # always signal to enter
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


def test_slippage_reduces_returns():
    model = DummyModel()
    df = _make_df()
    base_cfg = {"threshold": 0.5, "trailing_stop_pips": 1000, "order_size": 100}
    metrics_no_slip, _ = backtest.backtest_on_df(
        df, model, base_cfg, return_returns=True
    )

    cfg_slip = dict(base_cfg)
    cfg_slip["slippage_bps"] = 10
    metrics_slip, _ = backtest.backtest_on_df(
        df, model, cfg_slip, return_returns=True
    )

    assert metrics_slip["total_return"] < metrics_no_slip["total_return"]
    assert metrics_slip["skipped_trades"] == 0
    assert metrics_slip["partial_fills"] == 0

