"""Utilities for syncing state to a shared backend.

The backend location is configured via the ``STATE_BACKEND`` environment
variable.  When set, checkpoints and decision logs are copied to the backend
on update and can be pulled from the backend on startup.  The backend can be
an NFS mount, S3 bucket or any destination supported by ``rsync``.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

BACKEND = os.getenv("STATE_BACKEND")
LAST_SYNC: float = 0.0

logger = logging.getLogger(__name__)


def _run(cmd: list[str]) -> bool:
    """Run ``cmd`` and return ``True`` on success.

    Errors are logged and ``False`` is returned rather than raising so callers
    can decide how to handle failures.
    """
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("state sync command failed: %s", exc)
        return False


def _run_with_backoff(cmd: list[str], retries: int = 3, delay: float = 1.0) -> bool:
    """Run ``cmd`` with simple exponential backoff."""
    for _ in range(retries):
        if _run(cmd):
            return True
        time.sleep(delay)
        delay *= 2
    return False


def _sync_dir(src: Path, dst: str, backend: str) -> bool:
    if backend.startswith("s3://"):
        cmd = ["aws", "s3", "sync", str(src), dst]
    else:
        cmd = ["rsync", "-az", f"{src}/", dst]
    return _run_with_backoff(cmd)


def _sync_file(src: Path, dst: str, backend: str) -> bool:
    if backend.startswith("s3://"):
        cmd = ["aws", "s3", "cp", str(src), dst]
    else:
        cmd = ["rsync", "-az", str(src), dst]
    return _run_with_backoff(cmd)


def sync_checkpoints() -> bool:
    """Replicate local checkpoints to the backend."""
    if not BACKEND:
        return True
    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.exists():
        return True
    ok = _sync_dir(ckpt_dir, f"{BACKEND.rstrip('/')}/checkpoints", BACKEND)
    if ok:
        global LAST_SYNC
        LAST_SYNC = time.time()
        logger.info("Synced checkpoints to %s", BACKEND)
    else:
        logger.warning("Failed to sync checkpoints to %s", BACKEND)
    return ok


def sync_decisions() -> bool:
    """Replicate the decision log to the backend."""
    if not BACKEND:
        return True
    dec_log = Path("logs/decisions.parquet.enc")
    if not dec_log.exists():
        return True
    ok = _sync_file(dec_log, f"{BACKEND.rstrip('/')}/logs/decisions.parquet", BACKEND)
    if ok:
        global LAST_SYNC
        LAST_SYNC = time.time()
        logger.info("Synced decision log to %s", BACKEND)
    else:
        logger.warning("Failed to sync decision log to %s", BACKEND)
    return ok


def pull_checkpoints() -> None:
    """Retrieve checkpoints from the backend if available."""
    if not BACKEND:
        return
    dst = Path("checkpoints")
    dst.mkdir(exist_ok=True)
    src = f"{BACKEND.rstrip('/')}/checkpoints"
    if BACKEND.startswith("s3://"):
        _run_with_backoff(["aws", "s3", "sync", src, str(dst)])
    else:
        _run_with_backoff(["rsync", "-az", f"{src}/", str(dst)])


def pull_decisions() -> None:
    """Retrieve the decision log from the backend if available."""
    if not BACKEND:
        return
    dst = Path("logs")
    dst.mkdir(exist_ok=True)
    src = f"{BACKEND.rstrip('/')}/logs/decisions.parquet.enc"
    if BACKEND.startswith("s3://"):
        _run_with_backoff(["aws", "s3", "cp", src, str(dst / 'decisions.parquet.enc')])
    else:
        _run_with_backoff(["rsync", "-az", src, str(dst / 'decisions.parquet.enc')])


def sync_event_store(db_path: Path, dataset_dir: Path | None = None, backend: str | None = None) -> bool:
    """Replicate the event store database and dataset."""
    backend = backend or BACKEND
    if not backend:
        return True
    dst_root = f"{backend.rstrip('/')}/event_store"
    ok = _sync_file(db_path, f"{dst_root}/{db_path.name}", backend)
    if dataset_dir and Path(dataset_dir).exists():
        ok = _sync_dir(Path(dataset_dir), f"{dst_root}/dataset", backend) and ok
    if ok:
        global LAST_SYNC
        LAST_SYNC = time.time()
        logger.info("Synced event store to %s", backend)
    else:
        logger.warning("Failed to sync event store to %s", backend)
    return ok


def pull_event_store(db_path: Path | None = None, dataset_dir: Path | None = None, backend: str | None = None) -> None:
    """Retrieve the event store from the backend if available."""
    backend = backend or BACKEND
    if not backend:
        return
    db_path = Path(db_path or os.getenv("EVENT_STORE_PATH", "event_store/events.db"))
    dataset_dir = Path(dataset_dir or db_path.with_suffix(".parquet"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    root = f"{backend.rstrip('/')}/event_store"
    if backend.startswith("s3://"):
        _run_with_backoff(["aws", "s3", "cp", f"{root}/{db_path.name}", str(db_path)])
        _run_with_backoff(["aws", "s3", "sync", f"{root}/dataset", str(dataset_dir)])
    else:
        _run_with_backoff(["rsync", "-az", f"{root}/{db_path.name}", str(db_path)])
        _run_with_backoff(["rsync", "-az", f"{root}/dataset/", str(dataset_dir)])
    logger.info("Pulled event store from %s", backend)


def check_health(max_lag: int = 300) -> bool:
    """Return ``True`` if the last successful sync is within ``max_lag`` seconds."""
    if not BACKEND:
        return True
    return (time.time() - LAST_SYNC) <= max_lag
