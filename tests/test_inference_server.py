"""Unit tests for the lightweight inference server helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from services import inference_server


@dataclass
class _DummyTask:
    """Simple stand-in for ``asyncio.Task`` used by the watcher."""

    _done: bool = False
    _cancelled: bool = False
    cancel_calls: int = 0

    def cancel(self) -> None:
        self.cancel_calls += 1
        self._done = True
        self._cancelled = True

    def done(self) -> bool:  # pragma: no cover - trivial accessor
        return self._done

    def cancelled(self) -> bool:  # pragma: no cover - trivial accessor
        return self._cancelled


class _DummyMonitor:
    """Monitor stub that tracks subscriptions and tasks."""

    def __init__(self) -> None:
        self.queue_calls = 0
        self.create_task_calls = 0
        self.tasks: list[_DummyTask] = []

    def subscribe(self):  # type: ignore[no-untyped-def]
        self.queue_calls += 1
        return object()

    def create_task(self, coro):  # type: ignore[no-untyped-def]
        self.create_task_calls += 1
        coro.close()
        task = _DummyTask()
        self.tasks.append(task)
        return task

    def stop(self) -> None:  # pragma: no cover - not exercised here
        for task in list(self.tasks):
            task.cancel()
        self.tasks.clear()


@pytest.fixture(autouse=True)
def _reset_watcher(monkeypatch: pytest.MonkeyPatch):
    """Ensure tests always start with a clean watcher reference."""

    monkeypatch.setattr(inference_server, "_WIDTH_WATCH_TASK", None, raising=False)


def test_ensure_watcher_starts_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monitor = _DummyMonitor()
    monkeypatch.setattr(inference_server, "monitor", monitor, raising=False)

    inference_server._ensure_watcher()
    assert monitor.create_task_calls == 1

    # A second call reuses the active task without creating a duplicate watcher.
    inference_server._ensure_watcher()
    assert monitor.create_task_calls == 1


def test_ensure_watcher_recovers_after_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _DummyMonitor()
    monkeypatch.setattr(inference_server, "monitor", monitor, raising=False)

    # Simulate a cancelled watcher from a prior shutdown.
    inference_server._WIDTH_WATCH_TASK = _DummyTask(_done=True, _cancelled=True)
    inference_server._ensure_watcher()

    assert monitor.create_task_calls == 1
    assert isinstance(inference_server._WIDTH_WATCH_TASK, _DummyTask)
