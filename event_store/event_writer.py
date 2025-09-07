from __future__ import annotations

import datetime as _dt
import json
import threading
from pathlib import Path
from typing import Any, Dict
import os

_LOCK = threading.Lock()
_BASE_PATH = Path(os.getenv("EVENT_LOG_PATH", "event_store/events"))


def record(event_type: str, payload: Dict[str, Any], base_path: Path | str | None = None) -> None:
    """Append an event to partitioned Parquet files.

    Events are partitioned by ``type`` and ``date`` using the "hive" layout. The
    ``payload`` is serialised to JSON.  When :mod:`pyarrow` is unavailable the
    function is a no-op.
    """
    path = Path(base_path) if base_path is not None else _BASE_PATH
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.dataset as ds  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return

    ts = _dt.datetime.utcnow()
    data = json.dumps(payload, default=str)
    table = pa.table(
        {
            "timestamp": [ts],
            "type": [event_type],
            "date": [ts.date().isoformat()],
            "payload": [data],
        }
    )
    path.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        ds.write_dataset(
            table,
            base_dir=str(path),
            format="parquet",
            partitioning=["type", "date"],
            existing_data_behavior="overwrite_or_ignore",
            file_options=ds.ParquetFileFormat().make_write_options(compression="zstd"),
        )

