"""Tests covering the inference server shutdown behaviour."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import asyncio

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from services import inference_server


@dataclass
class _DummyTask:
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
    def __init__(self) -> None:
        self.create_task_calls = 0
        self.stop_calls = 0
        self.tasks: list[_DummyTask] = []
        self.all_tasks: list[_DummyTask] = []

    def subscribe(self):  # type: ignore[no-untyped-def]
        return object()

    def create_task(self, coro):  # type: ignore[no-untyped-def]
        self.create_task_calls += 1
        coro.close()
        task = _DummyTask()
        self.tasks.append(task)
        self.all_tasks.append(task)
        return task

    def stop(self) -> None:
        self.stop_calls += 1
        for task in list(self.tasks):
            task.cancel()
        self.tasks.clear()


class _DummyExecutor:
    def __init__(self) -> None:
        self.shutdown_calls: list[bool] = []

    def shutdown(self, wait: bool = True) -> None:
        self.shutdown_calls.append(wait)


def test_shutdown_cleans_up_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    monitor = _DummyMonitor()
    executor = _DummyExecutor()
    monkeypatch.setattr(inference_server, "monitor", monitor, raising=False)
    monkeypatch.setattr(inference_server, "_EXECUTOR", executor, raising=False)
    monkeypatch.setattr(inference_server, "_WIDTH_WATCH_TASK", None, raising=False)

    # Simulate application startup triggering the background watcher.
    inference_server._ensure_watcher()
    assert monitor.create_task_calls == 1
    assert monitor.tasks

    first_task = monitor.all_tasks[0]

    asyncio.run(inference_server.app.router.shutdown())

    assert monitor.stop_calls == 1
    assert not monitor.tasks
    assert first_task.cancelled()
    assert inference_server._WIDTH_WATCH_TASK is None
    assert executor.shutdown_calls == [False]


def test_shutdown_handles_missing_watcher(monkeypatch: pytest.MonkeyPatch) -> None:
    monitor = _DummyMonitor()
    executor = _DummyExecutor()
    monkeypatch.setattr(inference_server, "monitor", monitor, raising=False)
    monkeypatch.setattr(inference_server, "_EXECUTOR", executor, raising=False)
    monkeypatch.setattr(inference_server, "_WIDTH_WATCH_TASK", None, raising=False)

    asyncio.run(inference_server.app.router.shutdown())

    assert monitor.stop_calls == 1
    assert not monitor.tasks
    assert inference_server._WIDTH_WATCH_TASK is None
    assert executor.shutdown_calls == [False]
