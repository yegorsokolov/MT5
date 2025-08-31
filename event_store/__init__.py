from __future__ import annotations

import sqlite3
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import threading
import datetime as _dt
import os
import logging
import requests
from utils.resource_monitor import monitor

try:  # optional state sync
    from core import state_sync
except Exception:  # pragma: no cover - optional dependency
    state_sync = None


class EventStore:
    """Simple append-only event store backed by SQLite."""

    def __init__(
        self,
        path: str | Path | None = None,
        dataset_dir: str | Path | None = None,
        remote_url: str | None = None,
        disk_io_threshold: int = 100 * 1024 * 1024,
        backend: str | None = None,
    ) -> None:
        base = Path(__file__).resolve().parent
        self.path = Path(path) if path else base / "events.db"
        self.conn = sqlite3.connect(self.path)
        self.ds_path = Path(dataset_dir) if dataset_dir else self.path.with_suffix(".parquet")
        try:
            self.ds_path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self.remote_url = remote_url or os.getenv("EVENT_STORE_URL")
        self.disk_io_threshold = disk_io_threshold
        self.backend = backend or (getattr(state_sync, "BACKEND", None) if state_sync else None)
        self.logger = logging.getLogger(__name__)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        self.lock = threading.Lock()

    def record(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Record an event of the given type with the provided payload."""
        ts = _dt.datetime.utcnow().isoformat()
        data = json.dumps(payload, default=str)
        usage = getattr(monitor, "latest_usage", {})
        if self.remote_url and usage.get("disk_write", 0) > self.disk_io_threshold:
            try:
                requests.post(
                    self.remote_url,
                    json={"timestamp": ts, "type": event_type, "payload": payload},
                    timeout=5,
                )
                return
            except Exception:
                pass
        with self.lock:
            self.conn.execute(
                "INSERT INTO events(timestamp, type, payload) VALUES (?, ?, ?)",
                (ts, event_type, data),
            )
            self.conn.commit()
            try:
                import pyarrow as pa  # type: ignore
                import pyarrow.dataset as ds  # type: ignore

                ts_dt = _dt.datetime.fromisoformat(ts)
                table = pa.table(
                    {
                        "timestamp": [ts_dt],
                        "type": [event_type],
                        "date": [ts_dt.date().isoformat()],
                        "payload": [data],
                    }
                )
                ds.write_dataset(
                    table,
                    base_dir=str(self.ds_path),
                    format="parquet",
                    partitioning=["type", "date"],
                    existing_data_behavior="overwrite_or_ignore",
                    file_options=ds.ParquetFileFormat().make_write_options(
                        compression="zstd"
                    ),
                )
            except Exception:
                pass
            self._replicate()

    def iter_events(self, event_type: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        """Yield events in order, optionally filtered by type."""
        cur = self.conn.cursor()
        if event_type:
            cur.execute(
                "SELECT timestamp, type, payload FROM events WHERE type=? ORDER BY id",
                (event_type,),
            )
        else:
            cur.execute("SELECT timestamp, type, payload FROM events ORDER BY id")
        for ts, et, pl in cur.fetchall():
            yield {"timestamp": ts, "type": et, "payload": json.loads(pl)}

    def close(self) -> None:  # pragma: no cover - trivial
        self.conn.close()

    def _replicate(self) -> None:
        if not self.backend or not state_sync:
            return
        try:
            ok = state_sync.sync_event_store(self.path, self.ds_path, self.backend)
        except Exception:
            ok = False
        if ok:
            self.logger.info("Replicated event store to %s", self.backend)
        else:
            self.logger.warning("Event store replication to %s failed", self.backend)


__all__ = ["EventStore"]
