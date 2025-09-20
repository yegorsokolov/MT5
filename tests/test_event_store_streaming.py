import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest


class _InstrumentedCursor:
    def __init__(self, cursor: Any, stats: dict[str, int]) -> None:
        self._cursor = cursor
        self._stats = stats

    def execute(self, *args: Any, **kwargs: Any) -> "_InstrumentedCursor":
        self._cursor.execute(*args, **kwargs)
        return self

    def fetchmany(self, size: int | None = None):
        self._stats["fetchmany_calls"] += 1
        if size is None:
            return self._cursor.fetchmany()
        return self._cursor.fetchmany(size)

    def fetchall(self):
        self._stats["fetchall_calls"] += 1
        return self._cursor.fetchall()

    def fetchone(self):
        return self._cursor.fetchone()

    def close(self) -> None:
        self._cursor.close()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._cursor, item)


class _InstrumentedConnection:
    def __init__(self, conn: Any, stats: dict[str, int]) -> None:
        self._conn = conn
        self._stats = stats

    def cursor(self, *args: Any, **kwargs: Any) -> _InstrumentedCursor:
        return _InstrumentedCursor(self._conn.cursor(*args, **kwargs), self._stats)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._conn, item)


@pytest.mark.parametrize("event_count", [5000])
def test_iter_events_streams_in_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path, event_count: int
):
    sys.modules.pop("event_store", None)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))

    resource_monitor_stub = types.ModuleType("utils.resource_monitor")
    resource_monitor_stub.monitor = types.SimpleNamespace(latest_usage={})
    utils_stub = types.ModuleType("utils")
    utils_stub.__path__ = []
    utils_stub.resource_monitor = resource_monitor_stub

    monkeypatch.setitem(sys.modules, "utils", utils_stub)
    monkeypatch.setitem(sys.modules, "utils.resource_monitor", resource_monitor_stub)

    EventStore = importlib.import_module("event_store").EventStore

    store = EventStore(path=tmp_path / "events.db")
    try:
        rows = [
            (
                f"2024-01-01T00:00:00.{i:06d}",
                "stream",
                json.dumps({"idx": i}),
            )
            for i in range(event_count)
        ]
        with store.lock:
            cursor = store.conn.cursor()
            try:
                cursor.executemany(
                    "INSERT INTO events(timestamp, type, payload) VALUES (?, ?, ?)",
                    rows,
                )
                store.conn.commit()
            finally:
                cursor.close()

        stats = {"fetchmany_calls": 0, "fetchall_calls": 0}
        instrumented_conn = _InstrumentedConnection(store.conn, stats)
        monkeypatch.setattr(store, "conn", instrumented_conn)

        iterator = store.iter_events()
        first = next(iterator)

        assert stats["fetchall_calls"] == 0
        assert stats["fetchmany_calls"] == 1
        assert first["payload"]["idx"] == 0

        remaining = list(iterator)
        assert len(remaining) + 1 == event_count
        assert remaining[-1]["payload"]["idx"] == event_count - 1
        assert stats["fetchmany_calls"] >= 2
    finally:
        store.close()
