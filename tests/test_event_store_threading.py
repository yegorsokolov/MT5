import importlib
import sys
import threading
import types
from pathlib import Path

import pytest

if importlib.util.find_spec("pydantic") is None:
    utils_stub = types.ModuleType("utils")
    utils_stub.__path__ = []  # type: ignore[attr-defined]
    monitor_stub = types.SimpleNamespace(latest_usage={})
    resource_monitor_stub = types.ModuleType("utils.resource_monitor")
    resource_monitor_stub.monitor = monitor_stub
    sys.modules["utils"] = utils_stub
    sys.modules["utils.resource_monitor"] = resource_monitor_stub

repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

sys.modules.pop("scheduler", None)
sys.modules.pop("config_models", None)
sys.modules.pop("event_store", None)

from event_store import EventStore
import scheduler


def _reset_scheduler_state() -> None:
    """Ensure the scheduler globals are reset before running a test."""

    scheduler.stop_scheduler()
    scheduler._tasks.clear()
    scheduler._loop = None
    scheduler._thread = None
    scheduler._started = False
    scheduler._last_retrain_ts = None
    scheduler._failed_retrain_attempts.clear()
    scheduler._retrain_watcher = None


def test_retrain_watcher_consumes_events_from_background_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reset_scheduler_state()

    store = EventStore(tmp_path / "events.db")
    store.record("retrain", {"model": "classic"})

    processed = threading.Event()
    consumed: list[dict[str, object]] = []
    errors: list[Exception] = []

    def _fake_process(store_obj: EventStore) -> None:
        try:
            consumed.extend(store_obj.iter_events("retrain"))
        except Exception as exc:  # pragma: no cover - debugging aid
            errors.append(exc)
            raise
        finally:
            processed.set()

    monkeypatch.setattr(scheduler, "process_retrain_events", _fake_process)

    watcher = scheduler.subscribe_retrain_events(store=store, interval=0.01)
    assert watcher is not None
    try:
        assert processed.wait(timeout=2), "retrain watcher did not process events"
    finally:
        scheduler.stop_scheduler()
        store.close()

    assert not errors
    assert consumed and consumed[0]["payload"] == {"model": "classic"}
