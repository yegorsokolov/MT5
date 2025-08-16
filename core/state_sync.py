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


def _sync_dir(src: Path, dst: str) -> bool:
    if BACKEND.startswith("s3://"):
        return _run(["aws", "s3", "sync", str(src), dst])
    return _run(["rsync", "-az", f"{src}/", dst])


def _sync_file(src: Path, dst: str) -> bool:
    if BACKEND.startswith("s3://"):
        return _run(["aws", "s3", "cp", str(src), dst])
    return _run(["rsync", "-az", str(src), dst])


def sync_checkpoints() -> bool:
    """Replicate local checkpoints to the backend."""
    if not BACKEND:
        return True
    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.exists():
        return True
    ok = _sync_dir(ckpt_dir, f"{BACKEND.rstrip('/')}/checkpoints")
    if ok:
        global LAST_SYNC
        LAST_SYNC = time.time()
    return ok


def sync_decisions() -> bool:
    """Replicate the decision log to the backend."""
    if not BACKEND:
        return True
    dec_log = Path("logs/decisions.parquet")
    if not dec_log.exists():
        return True
    ok = _sync_file(dec_log, f"{BACKEND.rstrip('/')}/logs/decisions.parquet")
    if ok:
        global LAST_SYNC
        LAST_SYNC = time.time()
    return ok


def pull_checkpoints() -> None:
    """Retrieve checkpoints from the backend if available."""
    if not BACKEND:
        return
    dst = Path("checkpoints")
    dst.mkdir(exist_ok=True)
    src = f"{BACKEND.rstrip('/')}/checkpoints"
    if BACKEND.startswith("s3://"):
        _run(["aws", "s3", "sync", src, str(dst)])
    else:
        _run(["rsync", "-az", f"{src}/", str(dst)])


def pull_decisions() -> None:
    """Retrieve the decision log from the backend if available."""
    if not BACKEND:
        return
    dst = Path("logs")
    dst.mkdir(exist_ok=True)
    src = f"{BACKEND.rstrip('/')}/logs/decisions.parquet"
    if BACKEND.startswith("s3://"):
        _run(["aws", "s3", "cp", src, str(dst / 'decisions.parquet')])
    else:
        _run(["rsync", "-az", src, str(dst / 'decisions.parquet')])


def check_health(max_lag: int = 300) -> bool:
    """Return ``True`` if the last successful sync is within ``max_lag`` seconds."""
    if not BACKEND:
        return True
    return (time.time() - LAST_SYNC) <= max_lag
