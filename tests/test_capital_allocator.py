import sys
import types
from pathlib import Path

import pytest
import pandas as pd

# Stub optional dependencies used by ExecutionEngine
sys.modules.setdefault(
    "analysis.broker_tca", types.SimpleNamespace(broker_tca=types.SimpleNamespace(record=lambda *a, **k: None))
)
sys.modules.setdefault(
    "brokers.connection_manager",
    types.SimpleNamespace(get_active_broker=lambda: types.SimpleNamespace(__class__=type("B", (), {"__name__": "Dummy"}))),
)
sys.modules.setdefault(
    "metrics",
    types.SimpleNamespace(
        SLIPPAGE_BPS=types.SimpleNamespace(set=lambda *a, **k: None),
        REALIZED_SLIPPAGE_BPS=types.SimpleNamespace(set=lambda *a, **k: None),
        RECONNECT_COUNT=types.SimpleNamespace(inc=lambda *a, **k: None),
        FEATURE_ANOMALIES=types.SimpleNamespace(inc=lambda *a, **k: None),
        RESOURCE_RESTARTS=types.SimpleNamespace(inc=lambda *a, **k: None),
        QUEUE_DEPTH=types.SimpleNamespace(inc=lambda *a, **k: None),
        BATCH_LATENCY=types.SimpleNamespace(inc=lambda *a, **k: None),
        PRED_CACHE_HIT=types.SimpleNamespace(inc=lambda *a, **k: None),
        PRED_CACHE_HIT_RATIO=types.SimpleNamespace(inc=lambda *a, **k: None),
        PLUGIN_RELOADS=types.SimpleNamespace(inc=lambda *a, **k: None),
    ),
)
sys.modules.setdefault(
    "event_store.event_writer", types.SimpleNamespace(record=lambda *a, **k: None)
)
sys.modules.setdefault("event_store", types.SimpleNamespace())


class _Reg:
    def __init__(self, auto_refresh: bool = True):
        self._path = None

    def register_policy(self, name: str, path: Path, meta: dict) -> None:
        self._path = Path(path)

    def get_policy_path(self) -> Path:
        return self._path


sys.modules.setdefault("model_registry", types.SimpleNamespace(ModelRegistry=_Reg))
analytics_pkg = types.ModuleType("analytics")
analytics_pkg.__path__ = []
analytics_pkg.decision_logger = types.SimpleNamespace(log=lambda *a, **k: None)
sys.modules.setdefault("analytics", analytics_pkg)
sys.modules.setdefault(
    "analytics.metrics_aggregator",
    types.SimpleNamespace(record_metric=lambda *a, **k: None),
)
sys.modules.setdefault(
    "data.order_book",
    types.SimpleNamespace(
        load_order_book=lambda src: pd.DataFrame(src),
        compute_order_book_features=lambda df: pd.concat(
            [
                df.assign(
                    depth_imbalance=0.0,
                    vw_spread=0.0,
                    liquidity=0.0,
                    slippage=0.0,
                ),
                df.iloc[-1:].copy(),
                df.iloc[-1:].copy(),
                df.iloc[-1:].copy(),
            ],
            ignore_index=True,
        ),
    ),
)
utils_stub = types.ModuleType("utils")
utils_stub.load_config = lambda *a, **k: {}
sys.modules.setdefault("utils", utils_stub)
sys.modules.setdefault(
    "utils.resource_monitor",
    types.SimpleNamespace(
        ResourceMonitor=object,
        monitor=types.SimpleNamespace(capability_tier=lambda: "lite"),
    ),
)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.capital_allocator import CapitalAllocator
from execution.engine import ExecutionEngine


class DummyOpt:
    def get_params(self):
        return {"limit_offset": 0.0, "slice_size": None}

    def schedule_nightly(self):
        pass


def test_allocation_sums_to_one():
    allocator = CapitalAllocator()
    pnl = {"s1": 0.1, "s2": -0.2, "s3": 0.05}
    risk = {k: 0.1 for k in pnl}
    weights = allocator.allocate(pnl, risk)
    assert pytest.approx(sum(weights.values())) == 1.0


def test_allocator_responds_to_pnl():
    allocator = CapitalAllocator()
    risk = {"a": 0.1, "b": 0.1}
    w_good = allocator.allocate({"a": 0.2, "b": -0.1}, risk)
    w_bad = allocator.allocate({"a": -0.2, "b": 0.1}, risk)
    assert w_good["a"] > w_bad["a"]


def test_engine_rebalance_updates_weights():
    allocator = CapitalAllocator()
    engine = ExecutionEngine(optimizer=DummyOpt(), capital_allocator=allocator)
    risk = {"a": 0.1, "b": 0.1}
    w1 = engine.rebalance_capital({"a": 0.1, "b": 0.0}, risk)
    w2 = engine.rebalance_capital({"a": -0.1, "b": 0.2}, risk)
    assert pytest.approx(sum(w2.values())) == 1.0
    assert w1["a"] > w2["a"]
