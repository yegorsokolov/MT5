import asyncio
import importlib
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

pydantic_spec = importlib.util.find_spec("pydantic")
if pydantic_spec is None:
    pytest.skip("pydantic not available", allow_module_level=True)

repo_root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, repo_root)
sys.modules.pop("scheduler", None)
sys.modules.pop("config_models", None)
sys.modules.pop("event_store", None)
try:
    from mt5.config_models import AppConfig
except Exception:
    class _AppConfigDict(dict):
        def model_dump(self) -> dict[str, object]:
            return dict(self)

    class _AppConfig:
        @classmethod
        def model_validate(cls, data: dict[str, object]) -> _AppConfigDict:
            return _AppConfigDict(data)

    config_models_stub = types.ModuleType("config_models")
    config_models_stub.AppConfig = _AppConfig
    sys.modules["config_models"] = config_models_stub
    sys.modules.setdefault("mt5", types.ModuleType("mt5"))
    sys.modules["mt5.config_models"] = config_models_stub
from mt5.config_models import AppConfig
from mt5 import scheduler
from event_store import EventStore


def test_stop_scheduler_cancels_tasks_and_stops_loop():
    scheduler.stop_scheduler()
    scheduler._schedule_jobs([("dummy", True, lambda: None)])
    assert scheduler._thread is not None and scheduler._thread.is_alive()
    tasks = list(scheduler._tasks)
    scheduler.stop_scheduler()
    assert scheduler._loop is None
    assert scheduler._thread is None or not scheduler._thread.is_alive()
    assert scheduler._tasks == []
    assert all(t.cancelled() for t in tasks)


def test_start_stop_scheduler_clears_retrain_watcher(monkeypatch: pytest.MonkeyPatch):
    scheduler.stop_scheduler()
    scheduler._tasks.clear()
    scheduler._loop = None
    scheduler._thread = None
    scheduler._started = False
    scheduler._retrain_watcher = None
    scheduler._pending_retrain_models.clear()

    class DummyStore:
        def iter_events(self, event_type: str):  # pragma: no cover - signature compat
            assert event_type == "retrain"
            if False:
                yield None  # pragma: no cover - generator formality

    processed = threading.Event()

    async def fake_process(store_obj: DummyStore) -> None:
        processed.set()

    scheduler_flags = {
        "retrain_events": True,
        "resource_reprobe": False,
        "drift_detection": False,
        "feature_importance_drift": False,
        "change_point_detection": False,
        "checkpoint_cleanup": False,
        "trade_stats": False,
        "decision_review": False,
        "vacuum_history": False,
        "diagnostics": False,
        "backups": False,
        "regime_performance": False,
        "news_vector_store": False,
        "world_model_eval": False,
        "factor_update": False,
    }

    cfg = {"scheduler": scheduler_flags}

    monkeypatch.setattr(scheduler, "load_config", lambda: cfg)
    monkeypatch.setattr(scheduler, "process_retrain_events", fake_process)

    original_subscribe = scheduler.subscribe_retrain_events
    store = DummyStore()

    def subscribe_wrapper():
        return original_subscribe(store=store, interval=0.01)

    monkeypatch.setattr(scheduler, "subscribe_retrain_events", subscribe_wrapper)

    scheduler.start_scheduler()

    assert processed.wait(timeout=2), "retrain watcher never triggered"
    assert scheduler._retrain_watcher is not None

    loop = scheduler._loop
    assert loop is not None

    async def _pending_tasks() -> list[asyncio.Task]:
        current = asyncio.current_task()
        return [
            task
            for task in asyncio.all_tasks()
            if task is not current and not task.done()
        ]

    pending_tasks = asyncio.run_coroutine_threadsafe(_pending_tasks(), loop).result(
        timeout=1
    )
    assert pending_tasks, "expected retrain watcher task to be pending"

    watcher_future = scheduler._retrain_watcher
    initial_task_count = len(scheduler._tasks)

    scheduler._started = False
    scheduler.start_scheduler()

    assert scheduler._retrain_watcher is watcher_future
    assert len(scheduler._tasks) == initial_task_count

    scheduler.stop_scheduler()

    assert scheduler._retrain_watcher is None
    assert scheduler._tasks == []
    for task in pending_tasks:
        assert task.done()


def test_start_scheduler_respects_disabled_jobs(monkeypatch: pytest.MonkeyPatch):
    scheduler.stop_scheduler()
    scheduler._tasks.clear()
    scheduler._loop = None
    scheduler._thread = None
    scheduler._started = False
    scheduler._retrain_watcher = None
    scheduler._pending_retrain_models.clear()

    scheduler_flags = {
        "retrain_events": False,
        "resource_reprobe": False,
        "drift_detection": False,
        "feature_importance_drift": False,
        "change_point_detection": False,
        "checkpoint_cleanup": False,
        "trade_stats": False,
        "decision_review": False,
        "vacuum_history": False,
        "diagnostics": False,
        "backups": False,
        "regime_performance": False,
        "news_vector_store": False,
        "world_model_eval": False,
        "factor_update": False,
    }
    cfg = AppConfig.model_validate(
        {
            "strategy": {"symbols": ["TEST"], "risk_per_trade": 0.01},
            "scheduler": scheduler_flags,
        }
    )

    retrain_calls: list[None] = []

    def _fake_subscribe() -> None:
        retrain_calls.append(None)

    monkeypatch.setattr(scheduler, "load_config", lambda: cfg)
    monkeypatch.setattr(scheduler, "subscribe_retrain_events", _fake_subscribe)

    scheduler.start_scheduler()

    assert retrain_calls == []
    assert scheduler._tasks == []
    assert scheduler._loop is None
    scheduler._started = False


def test_subscribe_retrain_events_runs_from_plain_thread(
    monkeypatch: pytest.MonkeyPatch,
):
    scheduler.stop_scheduler()
    scheduler._tasks.clear()
    scheduler._loop = None
    scheduler._thread = None
    scheduler._started = False
    scheduler._last_retrain_ts = None
    scheduler._retrain_watcher = None
    scheduler._pending_retrain_models.clear()

    consumed: list[dict[str, object]] = []
    processed = threading.Event()

    class DummyStore:
        def __init__(self) -> None:
            self._events = [
                {
                    "timestamp": "2024-01-01T00:00:00",
                    "type": "retrain",
                    "payload": {"model": "classic"},
                }
            ]

        def iter_events(self, event_type: str):  # pragma: no cover - signature compat
            while self._events:
                yield self._events.pop(0)

    store = DummyStore()

    async def fake_process(store_obj: DummyStore) -> None:
        for event in store_obj.iter_events("retrain"):
            consumed.append(event)
            processed.set()

    monkeypatch.setattr(scheduler, "process_retrain_events", fake_process)

    thread = threading.Thread(
        target=scheduler.subscribe_retrain_events, args=(store, 0.01)
    )
    thread.start()
    thread.join()

    assert processed.wait(timeout=2), "retrain events were not processed"
    assert consumed == [
        {
            "timestamp": "2024-01-01T00:00:00",
            "type": "retrain",
            "payload": {"model": "classic"},
        }
    ]
    assert scheduler._tasks

    scheduler.stop_scheduler()


def test_process_retrain_events_retries_failed_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler.stop_scheduler()
    scheduler._tasks.clear()
    scheduler._loop = None
    scheduler._thread = None
    scheduler._started = False
    scheduler._last_retrain_ts = None
    scheduler._failed_retrain_attempts.clear()
    scheduler._retrain_watcher = None

    events = [
        {
            "timestamp": "2024-02-02T12:34:56",
            "type": "retrain",
            "payload": {"model": "classic"},
        }
    ]

    class DummyStore:
        def iter_events(self, event_type: str):  # pragma: no cover - signature compat
            assert event_type == "retrain"
            yield from events

    outcomes: list[tuple[str, str]] = []

    from analytics import metrics_store as metrics_module

    monkeypatch.setattr(
        metrics_module,
        "log_retrain_outcome",
        lambda model, status: outcomes.append((model, status)),
        raising=False,
    )

    call_count = {"value": 0}

    def fake_run(cmd, check=True, env=None):
        call_count["value"] += 1
        if call_count["value"] == 1:
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(scheduler.subprocess, "run", fake_run)

    store = DummyStore()

    asyncio.run(scheduler.process_retrain_events(store))

    attempt_key = "2024-02-02T12:34:56|classic"
    assert call_count["value"] == 1
    assert scheduler._last_retrain_ts is None
    assert scheduler._failed_retrain_attempts[attempt_key] == 1
    assert outcomes == [
        ("classic", "failed"),
        ("classic", "retry_scheduled"),
    ]

    asyncio.run(scheduler.process_retrain_events(store))

    assert call_count["value"] == 2
    assert scheduler._last_retrain_ts == "2024-02-02T12:34:56"
    assert scheduler._failed_retrain_attempts == {}
    assert outcomes == [
        ("classic", "failed"),
        ("classic", "retry_scheduled"),
        ("classic", "success"),
    ]

    scheduler._last_retrain_ts = None
    scheduler._failed_retrain_attempts.clear()


def test_slow_retrain_command_runs_off_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler.stop_scheduler()
    scheduler._tasks.clear()
    scheduler._loop = None
    scheduler._thread = None
    scheduler._started = False
    scheduler._last_retrain_ts = None
    scheduler._failed_retrain_attempts.clear()
    scheduler._retrain_watcher = None

    metrics_stub = types.SimpleNamespace(log_retrain_outcome=lambda *a, **k: None)
    analytics_pkg = types.SimpleNamespace(metrics_store=metrics_stub)
    monkeypatch.setitem(sys.modules, "analytics", analytics_pkg)
    monkeypatch.setitem(sys.modules, "analytics.metrics_store", metrics_stub)

    class DummyStore:
        def __init__(self) -> None:
            self._events = [
                {
                    "timestamp": "2024-03-03T00:00:00",
                    "type": "retrain",
                    "payload": {"model": "classic"},
                }
            ]

        def iter_events(self, event_type: str):  # pragma: no cover - signature compat
            assert event_type == "retrain"
            while self._events:
                yield self._events.pop(0)

    store = DummyStore()

    start_event = threading.Event()
    finish_event = threading.Event()

    def slow_run(cmd, check=True, env=None):
        start_event.set()
        time.sleep(0.1)
        finish_event.set()
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(scheduler.subprocess, "run", slow_run)

    progress_markers: list[float] = []

    async def _monitor_progress() -> None:
        while not start_event.is_set():
            await asyncio.sleep(0.005)
        loop = asyncio.get_running_loop()
        while not finish_event.is_set():
            progress_markers.append(loop.time())
            await asyncio.sleep(0.01)

    async def _run_test() -> None:
        monitor_task = asyncio.create_task(_monitor_progress())
        await scheduler.process_retrain_events(store)
        await asyncio.wait_for(monitor_task, timeout=1)

    asyncio.run(_run_test())

    assert start_event.is_set()
    assert finish_event.is_set()
    assert len(progress_markers) >= 1, "event loop stalled while retrain subprocess ran"
    assert scheduler._failed_retrain_attempts == {}
    assert scheduler._last_retrain_ts == "2024-03-03T00:00:00"


def test_schedule_retrain_skips_duplicates(tmp_path: Path) -> None:
    scheduler.stop_scheduler()
    scheduler._pending_retrain_models.clear()
    store = EventStore(tmp_path / "events.db")
    try:
        scheduler.schedule_retrain(model="classic", store=store)
        scheduler.schedule_retrain(model="classic", store=store)
        events = list(store.iter_events("retrain"))
        assert len(events) == 1
        scheduler._pending_retrain_models.clear()
        scheduler.schedule_retrain(model="classic", store=store)
        events = list(store.iter_events("retrain"))
        assert len(events) == 2
        scheduler.schedule_retrain(model="classic", store=store, force=True)
        events = list(store.iter_events("retrain"))
        assert len(events) == 3
    finally:
        store.close()


def test_process_retrain_events_clears_pending_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler.stop_scheduler()
    scheduler._pending_retrain_models.clear()
    scheduler._failed_retrain_attempts.clear()
    scheduler._last_retrain_ts = None

    class DummyStore:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def record(self, event_type: str, payload: dict[str, object]) -> None:
            assert event_type == "retrain"
            self.events.append(
                {
                    "timestamp": f"2024-05-01T00:00:0{len(self.events)}",
                    "type": event_type,
                    "payload": payload,
                }
            )

        def iter_events(self, event_type: str):  # pragma: no cover - signature compat
            assert event_type == "retrain"
            while self.events:
                yield self.events.pop(0)

    store = DummyStore()

    metrics_stub = types.SimpleNamespace(log_retrain_outcome=lambda *a, **k: None)
    analytics_pkg = types.SimpleNamespace(metrics_store=metrics_stub)
    monkeypatch.setitem(sys.modules, "analytics", analytics_pkg)
    monkeypatch.setitem(sys.modules, "analytics.metrics_store", metrics_stub)

    scheduler.schedule_retrain(model="classic", store=store)
    assert scheduler._pending_retrain_models == {"classic"}

    monkeypatch.setattr(
        scheduler,
        "_run_retrain_command",
        lambda *a, **k: scheduler._RetrainExecutionResult(True, object(), None),
    )

    asyncio.run(scheduler.process_retrain_events(store))

    assert scheduler._pending_retrain_models == set()
    assert scheduler._last_retrain_ts == "2024-05-01T00:00:00"

    scheduler.schedule_retrain(model="classic", store=store)
    assert scheduler._pending_retrain_models == {"classic"}

    asyncio.run(scheduler.process_retrain_events(store))
    assert scheduler._pending_retrain_models == set()

