import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

repo_root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, repo_root)
sys.modules.pop("scheduler", None)
sys.modules.pop("config_models", None)
sys.modules.pop("event_store", None)
from config_models import AppConfig
import scheduler


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


def test_start_scheduler_respects_disabled_jobs(monkeypatch: pytest.MonkeyPatch):
    scheduler.stop_scheduler()
    scheduler._tasks.clear()
    scheduler._loop = None
    scheduler._thread = None
    scheduler._started = False

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

    def fake_process(store_obj: DummyStore) -> None:
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
