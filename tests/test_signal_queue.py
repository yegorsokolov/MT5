import asyncio
from pathlib import Path
import sys
import types
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

_event_writer_stub = types.ModuleType("event_writer")
_EVENTS: list[tuple[str, dict, object | None]] = []


def _record_stub(event_type: str, payload: dict, base_path=None) -> None:
    _EVENTS.append((event_type, payload, base_path))


_event_writer_stub.record = _record_stub
_event_store_pkg = types.ModuleType("event_store")
_event_store_pkg.__path__ = []  # type: ignore[attr-defined]
_event_store_pkg.event_writer = _event_writer_stub  # type: ignore[attr-defined]
sys.modules.setdefault("event_store", _event_store_pkg)
sys.modules.setdefault("event_store.event_writer", _event_writer_stub)

_analysis_pkg = types.ModuleType("analysis")
_analysis_pkg.__path__ = []  # type: ignore[attr-defined]

_pipeline_stub = types.ModuleType("pipeline_anomaly")


def _validate_stub(_: object) -> bool:
    return True


_pipeline_stub.validate = _validate_stub  # type: ignore[attr-defined]
sys.modules.setdefault("analysis", _analysis_pkg)
sys.modules.setdefault("analysis.pipeline_anomaly", _pipeline_stub)
_analysis_pkg.pipeline_anomaly = _pipeline_stub  # type: ignore[attr-defined]

_models_pkg = types.ModuleType("models")
_models_pkg.__path__ = []  # type: ignore[attr-defined]

_conformal_stub = types.ModuleType("conformal")


def _coverage_stub(*_: object) -> float:
    return 1.0


_conformal_stub.evaluate_coverage = _coverage_stub  # type: ignore[attr-defined]
sys.modules.setdefault("models", _models_pkg)
sys.modules.setdefault("models.conformal", _conformal_stub)
_models_pkg.conformal = _conformal_stub  # type: ignore[attr-defined]

_residual_stub = types.ModuleType("residual_stacker")


def _load_stub(name: str) -> None:
    return None


def _predict_stub(features, base, model):  # noqa: ANN001 - signature mimics real module
    return [0.0 for _ in range(len(base))]


_residual_stub.load = _load_stub  # type: ignore[attr-defined]
_residual_stub.predict = _predict_stub  # type: ignore[attr-defined]
sys.modules.setdefault("models.residual_stacker", _residual_stub)
_models_pkg.residual_stacker = _residual_stub  # type: ignore[attr-defined]

_risk_manager_stub = types.ModuleType("risk_manager")


class _RiskManagerStub:
    def __init__(self, max_drawdown: float | None = None, **_: object) -> None:
        self.max_drawdown = max_drawdown

    def adjust_size(
        self,
        symbol: str,
        size: float,
        timestamp,
        direction: int,
        cred_interval=None,
    ) -> float:
        if cred_interval is None:
            return float(size) * float(direction)
        lower, upper = cred_interval
        width = float(abs(upper - lower))
        factor = max(0.0, 1.0 - min(width, 1.0))
        return float(size) * factor * float(direction)


_risk_manager_stub.RiskManager = _RiskManagerStub  # type: ignore[attr-defined]
sys.modules.setdefault("risk_manager", _risk_manager_stub)

import signal_queue
import scheduler as scheduler_mod

scheduler_mod.start_scheduler = lambda: None
import risk_manager


class _DummyBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    async def publish(self, topic: str, message: dict) -> None:
        self.published.append((topic, message))


def _stub_event(*_: object, **__: object) -> None:
    return None


def test_publish_dataframe_sync_runs_coroutine(monkeypatch):
    bus = _DummyBus()
    df = pd.DataFrame({"Timestamp": ["2024"], "prob": [0.55]})

    monkeypatch.setattr(signal_queue, "record_event", _stub_event)
    monkeypatch.setattr(signal_queue.pipeline_anomaly, "validate", _validate_stub)

    called = {"run": False}

    def fake_run(coro):
        called["run"] = True
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(signal_queue.asyncio, "run", fake_run)

    signal_queue.publish_dataframe(bus, df)

    assert called["run"] is True
    assert bus.published == [(signal_queue.Topics.SIGNALS, {"Timestamp": "2024", "prob": 0.55})]


def test_publish_dataframe_running_loop_schedules_task(monkeypatch):
    bus = _DummyBus()
    df = pd.DataFrame({"Timestamp": ["2025"], "prob": [0.42]})

    monkeypatch.setattr(signal_queue, "record_event", _stub_event)
    monkeypatch.setattr(signal_queue.pipeline_anomaly, "validate", _validate_stub)

    orig_run = asyncio.run

    def forbid_run(*args, **kwargs):
        raise AssertionError("asyncio.run should not be called when loop is running")

    monkeypatch.setattr(signal_queue.asyncio, "run", forbid_run)

    scheduled = []

    async def runner() -> None:
        loop = asyncio.get_running_loop()
        original_create_task = loop.create_task

        def capture(coro):
            task = original_create_task(coro)
            scheduled.append(task)
            return task

        monkeypatch.setattr(loop, "create_task", capture)

        signal_queue.publish_dataframe(bus, df)
        assert scheduled, "publish_dataframe should schedule the coroutine on the running loop"
        await asyncio.gather(*scheduled)

    orig_run(runner())

    assert bus.published == [(signal_queue.Topics.SIGNALS, {"Timestamp": "2025", "prob": 0.42})]


def test_publish_and_iter():
    queue = signal_queue.get_signal_backend({})
    df = pd.DataFrame({"Timestamp": ["2020"], "prob": [0.8], "confidence": [0.9]})
    signal_queue.publish_dataframe(queue, df)
    gen = signal_queue.iter_messages(queue)
    msg = asyncio.run(gen.__anext__())
    assert msg["prob"] == 0.8
    assert pytest.approx(msg["confidence"]) == 0.9


def test_async_publish_and_iter():
    queue = signal_queue.get_signal_backend({})
    df = pd.DataFrame({"Timestamp": ["2021"], "prob": [0.7], "confidence": [0.8]})
    asyncio.run(signal_queue.publish_dataframe_async(queue, df))
    gen = signal_queue.iter_messages(queue)
    msg = asyncio.run(asyncio.wait_for(gen.__anext__(), timeout=1))
    assert msg["prob"] == 0.7
    assert pytest.approx(msg["confidence"]) == 0.8


def test_publish_inside_running_loop():
    queue = signal_queue.get_signal_backend({})
    df = pd.DataFrame({"Timestamp": ["2023"], "prob": [0.6], "confidence": [0.7]})

    async def runner():
        signal_queue.publish_dataframe(queue, df)
        gen = signal_queue.iter_messages(queue)
        msg = await asyncio.wait_for(gen.__anext__(), timeout=1)
        assert msg["prob"] == 0.6
        assert pytest.approx(msg["confidence"]) == 0.7

    asyncio.run(runner())


def test_credible_interval_adjust_size():
    queue = signal_queue.get_signal_backend({})
    df = pd.DataFrame(
        {
            "Timestamp": ["2022"],
            "pred": [0.05],
            "ci_lower": [0.01],
            "ci_upper": [0.09],
        }
    )
    signal_queue.publish_dataframe(queue, df)
    gen = signal_queue.iter_messages(queue)
    msg = asyncio.run(gen.__anext__())
    assert "prediction" in msg
    ci = (msg["prediction"]["lower"], msg["prediction"]["upper"])
    rm = risk_manager.RiskManager(max_drawdown=1.0)
    sized = rm.adjust_size("EURUSD", 1.0, pd.Timestamp("2022"), 1, cred_interval=ci)
    assert sized < 1.0
