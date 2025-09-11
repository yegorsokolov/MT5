import asyncio
import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault(
    "analysis.broker_tca", types.SimpleNamespace(broker_tca=types.SimpleNamespace(record=lambda *a, **k: None))
)
sys.modules.setdefault(
    "brokers.connection_manager", types.SimpleNamespace(get_active_broker=lambda: types.SimpleNamespace(__class__=type("B", (), {"__name__": "Dummy"})))
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


@pytest.mark.asyncio
async def test_async_place_order_partial_fill_and_slippage():
    class DummyOpt:
        def get_params(self):
            return {"limit_offset": 0.0, "slice_size": None}

        def schedule_nightly(self):
            pass

    engine = ExecutionEngine(optimizer=DummyOpt())
    depth = {"bid_vol": 5.0, "ask_vol": 5.0}

    # ensure TWAP produces multiple slices
    engine.record_volume(1.0)
    engine.record_volume(1.0)

    def depth_cb():
        return depth["bid_vol"], depth["ask_vol"]

    async def modify_depth():
        # wait until first fill event queued then remove liquidity
        while engine.event_queue.empty():
            await asyncio.sleep(0)
        depth["ask_vol"] = 0.0

    asyncio.create_task(modify_depth())

    result = await engine.place_order(
        side="buy",
        quantity=10.0,
        bid=99.0,
        ask=101.0,
        bid_vol=depth["bid_vol"],
        ask_vol=depth["ask_vol"],
        mid=100.0,
        strategy="twap",
        expected_slippage_bps=10.0,
        depth_cb=depth_cb,
    )

    assert result["filled"] == pytest.approx(5.0)
    expected_price = 101.0 * (1 + 10.0 / 10000.0)
    assert result["avg_price"] == pytest.approx(expected_price)
    events = []
    while not engine.event_queue.empty():
        events.append(engine.event_queue.get_nowait())
    assert events[0]["type"] == "fill"
    assert events[0]["qty"] == pytest.approx(5.0)
    assert events[1]["type"] == "cancel"
    slippage = (result["avg_price"] - 100.0) / 100.0 * 10000.0
    assert slippage == pytest.approx(110.0, rel=1e-3)
