from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Callable, Iterable

try:
    from utils import load_config
except Exception:  # pragma: no cover - config loading optional
    load_config = lambda: {}

logger = logging.getLogger(__name__)

_loop: asyncio.AbstractEventLoop | None = None
_started = False
_tasks: list[asyncio.Future] = []

async def _runner(name: str, interval: float, func: Callable[[], None | asyncio.Future]) -> None:
    # delay first run by ``interval`` to avoid heavy startup work
    await asyncio.sleep(interval)
    while True:
        logger.info("Executing scheduled job: %s", name)
        try:
            result = func()
            if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
                await result
        except Exception:
            logger.exception("Job %s failed", name)
        await asyncio.sleep(interval)

def _schedule_jobs(jobs: Iterable[tuple[str, Callable[[], None | asyncio.Future]]]) -> None:
    global _loop
    if _loop is None:
        _loop = asyncio.new_event_loop()
        threading.Thread(target=_loop.run_forever, daemon=True).start()
    for name, func in jobs:
        task = asyncio.run_coroutine_threadsafe(_runner(name, 24 * 60 * 60, func), _loop)
        _tasks.append(task)
        logger.info("Scheduled job: %s", name)

def cleanup_checkpoints() -> None:
    """Remove old checkpoints, keeping the most recent files."""
    path = Path(os.getenv("CHECKPOINT_DIR", "checkpoints"))
    keep = int(os.getenv("KEEP_CHECKPOINTS", "5"))
    if not path.exists():
        return
    checkpoints = sorted(
        path.glob("checkpoint_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for ckpt in checkpoints[keep:]:
        try:
            ckpt.unlink()
            logger.info("Removed old checkpoint %s", ckpt)
        except Exception:
            logger.exception("Failed removing checkpoint %s", ckpt)

def resource_reprobe() -> None:
    """Refresh resource capability information."""
    try:
        from utils.resource_monitor import monitor

        monitor.capabilities = monitor._probe()
        logger.info("Refreshed resource capabilities: %s", monitor.capabilities)
    except Exception:
        logger.exception("Resource reprobe failed")

def run_drift_detection() -> None:
    """Run model/data drift comparison."""
    try:
        from monitor_drift import monitor

        monitor.compare()
        logger.info("Drift detection completed")
    except Exception:
        logger.exception("Drift detection failed")

def start_scheduler() -> None:
    """Start background scheduler based on configuration."""
    global _started
    if _started:
        return
    cfg = load_config()
    s_cfg = cfg.get("scheduler", {}) if isinstance(cfg, dict) else {}
    jobs: list[tuple[str, Callable[[], None | asyncio.Future]]] = []
    if s_cfg.get("resource_reprobe", True):
        jobs.append(("resource_reprobe", resource_reprobe))
    if s_cfg.get("drift_detection", True):
        jobs.append(("drift_detection", run_drift_detection))
    if s_cfg.get("checkpoint_cleanup", True):
        jobs.append(("checkpoint_cleanup", cleanup_checkpoints))
    if jobs:
        _schedule_jobs(jobs)
    _started = True
    logger.info("Scheduler started with %d job(s)", len(jobs))

__all__ = [
    "start_scheduler",
    "cleanup_checkpoints",
    "resource_reprobe",
    "run_drift_detection",
]
