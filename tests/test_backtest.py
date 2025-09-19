import pytest
import sys
import types
import logging
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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
    "analysis.strategy_evaluator",
    types.SimpleNamespace(StrategyEvaluator=object),
)


class _DummyExecutionEngine:
    def __init__(self, *args, **kwargs):
        pass

    def start_optimizer(self):
        return None

    def stop_optimizer(self):
        return None

    def execute(self, *args, **kwargs):
        return types.SimpleNamespace(status="filled", price=1.0)


sys.modules.setdefault(
    "execution",
    types.SimpleNamespace(ExecutionEngine=_DummyExecutionEngine),
)
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
        OptimizationLoopHandle=object,
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
sys.modules.setdefault(
    "crypto_utils",
    types.SimpleNamespace(
        _load_key=lambda *a, **k: None,
        encrypt=lambda *a, **k: b"",
        decrypt=lambda *a, **k: b"",
    ),
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
        )
    ),
)
sys.modules["requests"] = types.SimpleNamespace(
    Session=lambda: types.SimpleNamespace(post=lambda *a, **k: None),
    get=lambda *a, **k: None,
)
ray_stub = types.SimpleNamespace(remote=lambda f: f)
sys.modules.setdefault(
    "ray_utils",
    types.SimpleNamespace(ray=ray_stub, init=lambda **k: None, shutdown=lambda: None),
)

# Load backtest module with dummy logging to avoid import side effects
spec = importlib.util.spec_from_file_location(
    "backtest", Path(__file__).resolve().parents[1] / "backtest.py"
)
sys.modules["lightgbm"] = types.SimpleNamespace(LGBMClassifier=object)
backtest = importlib.util.module_from_spec(spec)
backtest.setup_logging = lambda: None
sys.modules["log_utils"] = types.SimpleNamespace(
    setup_logging=lambda: None, log_exceptions=lambda f: f
)
spec.loader.exec_module(backtest)

backtest.init_logging = lambda: logging.getLogger("test_backtest")

trailing_stop = backtest.trailing_stop


def test_trailing_stop_tightens():
    stop = trailing_stop(1.0, 1.05, 0.99, 0.01)
    assert stop == pytest.approx(1.04)
    stop = trailing_stop(1.0, 1.07, stop, 0.01)
    assert stop == pytest.approx(1.06)


def test_trailing_stop_does_not_loosen():
    stop = 1.04
    new_stop = trailing_stop(1.0, 1.05, stop, 0.02)
    assert new_stop == stop


def test_compute_metrics_constant_returns():
    returns = pd.Series([0.0] * 10)
    metrics = backtest.compute_metrics(returns)
    assert metrics["sharpe"] == 0.0
    assert all(np.isfinite(list(metrics.values())))


def test_compute_metrics_near_constant_positive_returns():
    returns = pd.Series([0.01] * 10)
    metrics = backtest.compute_metrics(returns)
    assert metrics["sharpe"] == 0.0
    assert all(np.isfinite(list(metrics.values())))
