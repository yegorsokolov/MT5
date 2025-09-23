import asyncio
import sys
import types
import importlib.util
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.slippage_model import SlippageModel

# minimal stubs for optional dependencies
class DummyGauge:
    def __init__(self, *a, **k):
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
    types.SimpleNamespace(monitor=types.SimpleNamespace(capability_tier=lambda: "lite")),
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
    ts = pd.date_range("2020-01-01", periods=7, freq="ms")
    data = {
        "Timestamp": ts,
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


def _run_naive_backtest(df: pd.DataFrame, model, cfg: dict, *, slippage_model=None):
    feats = backtest.feature_columns(df)
    probs = model.predict_proba(df[feats])[:, 1]

    threshold = cfg.get("threshold", 0.55)
    regime_thresholds = getattr(model, "regime_thresholds", {})
    distance = cfg.get("trailing_stop_pips", 20) * 1e-4
    order_size = cfg.get("order_size", 1.0)

    engine = backtest.ExecutionEngine()
    try:
        engine.start_optimizer()
        tier = getattr(backtest.monitor, "capability_tier", lambda: "lite")
        strategy = "ioc" if tier == "lite" else cfg.get("execution_strategy", "vwap")

        in_position = False
        entry = 0.0
        stop = 0.0
        returns: list[float] = []
        skipped = 0
        partial = 0

        delay_idx = np.arange(len(df))

        for i, (row, prob) in enumerate(zip(df.itertuples(index=False), probs)):
            delayed = df.iloc[delay_idx[i]]
            price_mid = getattr(delayed, "mid")
            bid = getattr(delayed, "Bid", price_mid)
            ask = getattr(delayed, "Ask", price_mid)
            bid_vol = getattr(delayed, "BidVolume", np.inf)
            ask_vol = getattr(delayed, "AskVolume", np.inf)

            engine.record_volume(bid_vol + ask_vol)
            regime = getattr(row, "market_regime", None)
            thr = (
                regime_thresholds.get(int(regime), threshold)
                if regime is not None
                else threshold
            )

            if not in_position and prob > thr:
                result = asyncio.run(
                    engine.place_order(
                        side="buy",
                        quantity=order_size,
                        bid=bid,
                        ask=ask,
                        bid_vol=bid_vol,
                        ask_vol=ask_vol,
                        mid=price_mid,
                        strategy=strategy,
                        expected_slippage_bps=cfg.get("slippage_bps", 0.0),
                        slippage_model=slippage_model,
                    )
                )
                if result["filled"] < order_size:
                    skipped += 1
                    continue
                in_position = True
                entry = result["avg_price"]
                stop = entry - distance
                continue

            if in_position:
                current_mid = getattr(row, "mid")
                stop = backtest.trailing_stop(entry, current_mid, stop, distance)
                if current_mid <= stop:
                    result = asyncio.run(
                        engine.place_order(
                            side="sell",
                            quantity=order_size,
                            bid=bid,
                            ask=ask,
                            bid_vol=bid_vol,
                            ask_vol=ask_vol,
                            mid=price_mid,
                            strategy=strategy,
                            expected_slippage_bps=cfg.get("slippage_bps", 0.0),
                            slippage_model=slippage_model,
                        )
                    )
                    fill_frac = min(result["filled"] / order_size, 1.0)
                    if fill_frac < 1.0:
                        partial += 1
                    exit_price = result["avg_price"]
                    returns.append(((exit_price - entry) / entry) * fill_frac)
                    in_position = False

        return returns, skipped, partial
    finally:
        engine.stop_optimizer()


def test_pnl_drops_with_latency_and_slippage():
    model = DummyModel()
    df = _make_df()
    cfg = {"threshold": 0.5, "trailing_stop_pips": 1000, "order_size": 100}
    metrics_base, _ = backtest.backtest_on_df(df, model, cfg, return_returns=True)

    metrics_latency, _ = backtest.backtest_on_df(
        df, model, cfg, latency_ms=1, return_returns=True
    )

    snaps_buy = [[(row.Ask, 10), (row.Ask + 1.0, 10)] for row in df.itertuples()]
    snaps_sell = [[(row.Bid, 10), (row.Bid - 1.0, 10)] for row in df.itertuples()]
    model_buy = SlippageModel(snaps_buy)
    model_sell = SlippageModel(snaps_sell)

    def slip(order_size, side):
        return model_buy(order_size, side) if side == "buy" else model_sell(order_size, side)

    metrics_slip, _ = backtest.backtest_on_df(
        df, model, cfg, slippage_model=slip, return_returns=True
    )

    assert metrics_latency["total_return"] < metrics_base["total_return"]
    assert metrics_slip["total_return"] < metrics_base["total_return"]


def test_backtest_single_event_loop_is_faster_than_naive():
    model = DummyModel()
    base = _make_df()
    df = pd.concat([base] * 150, ignore_index=True)
    df["Timestamp"] = pd.date_range("2020-01-01", periods=len(df), freq="ms")

    cfg = {"threshold": 0.5, "trailing_stop_pips": 1000, "order_size": 100}

    naive_times = []
    naive_returns = None
    for _ in range(2):
        start = perf_counter()
        naive_returns, _, _ = _run_naive_backtest(df, model, cfg)
        naive_times.append(perf_counter() - start)

    improved_times = []
    improved_returns = None
    for _ in range(2):
        start = perf_counter()
        _, improved_returns = backtest.backtest_on_df(
            df, model, cfg, return_returns=True
        )
        improved_times.append(perf_counter() - start)

    assert improved_returns is not None
    assert naive_returns is not None
    assert np.allclose(improved_returns.to_numpy(), np.array(naive_returns))

    naive_best = min(naive_times)
    improved_best = min(improved_times)
    assert improved_best < naive_best
    assert improved_best <= naive_best * 0.9
