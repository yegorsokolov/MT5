import sys
import types
from pathlib import Path

import pytest

# Stub optional dependencies
sys.modules.setdefault(
    "analysis.broker_tca",
    types.SimpleNamespace(broker_tca=types.SimpleNamespace(record=lambda *a, **k: None)),
)
sys.modules.setdefault(
    "brokers.connection_manager",
    types.SimpleNamespace(
        get_active_broker=lambda: types.SimpleNamespace(__class__=type("B", (), {"__name__": "Dummy"}))
    ),
)
sys.modules.setdefault(
    "metrics",
    types.SimpleNamespace(
        SLIPPAGE_BPS=types.SimpleNamespace(set=lambda *a, **k: None),
        REALIZED_SLIPPAGE_BPS=types.SimpleNamespace(set=lambda *a, **k: None),
    ),
)
sys.modules.setdefault(
    "event_store.event_writer", types.SimpleNamespace(record=lambda *a, **k: None)
)
sys.modules.setdefault("event_store", types.SimpleNamespace())
sys.modules.setdefault("model_registry", types.SimpleNamespace(ModelRegistry=object))
sys.modules.setdefault(
    "utils.resource_monitor",
    types.SimpleNamespace(monitor=types.SimpleNamespace(capability_tier=lambda: "lite")),
)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from execution.engine import ExecutionEngine


class DummyOpt:
    def get_params(self):
        return {"limit_offset": 0.0, "slice_size": None}

    def schedule_nightly(self):
        pass


@pytest.mark.asyncio
async def test_limit_order_full_fill():
    engine = ExecutionEngine(optimizer=DummyOpt())
    result = await engine.place_order(
        side="buy",
        quantity=5.0,
        bid=99.0,
        ask=101.0,
        bid_vol=5.0,
        ask_vol=10.0,
        mid=100.0,
        strategy="limit",
        limit_price=102.0,
    )
    assert result["filled"] == pytest.approx(5.0)
    assert result["avg_price"] == pytest.approx(102.0)
    events = []
    while not engine.event_queue.empty():
        events.append(engine.event_queue.get_nowait())
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "fill"
    assert e["qty"] == pytest.approx(5.0)
    assert e["price"] == pytest.approx(102.0)
    assert not e["partial"]


@pytest.mark.asyncio
async def test_limit_order_partial_fill():
    engine = ExecutionEngine(optimizer=DummyOpt())
    result = await engine.place_order(
        side="buy",
        quantity=10.0,
        bid=99.0,
        ask=101.0,
        bid_vol=5.0,
        ask_vol=5.0,
        mid=100.0,
        strategy="limit",
        limit_price=102.0,
    )
    assert result["filled"] == pytest.approx(5.0)
    events = []
    while not engine.event_queue.empty():
        events.append(engine.event_queue.get_nowait())
    assert events[0]["type"] == "fill"
    assert events[0]["qty"] == pytest.approx(5.0)
    assert events[0]["partial"]
    assert events[1]["type"] == "cancel"
    assert events[1]["qty"] == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_limit_order_expiration():
    engine = ExecutionEngine(optimizer=DummyOpt())
    result = await engine.place_order(
        side="buy",
        quantity=5.0,
        bid=99.0,
        ask=101.0,
        bid_vol=5.0,
        ask_vol=5.0,
        mid=100.0,
        strategy="limit",
        limit_price=100.0,
    )
    assert result["filled"] == pytest.approx(0.0)
    events = []
    while not engine.event_queue.empty():
        events.append(engine.event_queue.get_nowait())
    assert len(events) == 1
    assert events[0]["type"] == "cancel"
    assert events[0]["qty"] == pytest.approx(5.0)
