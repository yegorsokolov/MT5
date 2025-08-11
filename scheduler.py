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


def run_change_point_detection() -> None:
    """Run change point detection on recorded feature data."""
    try:
        import pandas as pd
        from analysis.change_point import ChangePointDetector
        from monitor_drift import DRIFT_METRICS
    except Exception:  # pragma: no cover - optional dependency
        logger.exception("Change point detection dependencies unavailable")
        return

    if not DRIFT_METRICS.exists():
        logger.info("No feature data for change point detection at %s", DRIFT_METRICS)
        return

    try:
        df = pd.read_parquet(DRIFT_METRICS)
    except Exception:
        logger.exception("Failed loading feature data from %s", DRIFT_METRICS)
        return

    detector = ChangePointDetector()
    cps = detector.record(df)
    if any(cps.values()):
        logger.warning("Change points detected: %s", cps)
    else:
        logger.info("No change points detected")


def run_trade_analysis() -> None:
    """Run trade analysis and persist daily statistics."""
    try:
        import pandas as pd
        import json
        from analytics.trade_analyzer import TradeAnalyzer
    except Exception:  # pragma: no cover - optional dependency
        logger.exception("Trade analysis dependencies unavailable")
        return

    trades_path = Path("reports/trades.csv")
    if not trades_path.exists():
        logger.info("No trades found at %s", trades_path)
        return

    try:
        trades = pd.read_csv(trades_path, parse_dates=["entry_time", "exit_time"])
        analyzer = TradeAnalyzer(trades)
        stats = analyzer.summary()
        out_dir = Path("reports/trade_stats")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{pd.Timestamp.utcnow():%Y-%m-%d}.json"
        with open(out_file, "w") as fh:
            json.dump(stats, fh)
        logger.info("Saved trade stats to %s", out_file)
    except Exception:
        logger.exception("Trade analysis failed")

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
    if s_cfg.get("change_point_detection", True):
        jobs.append(("change_point_detection", run_change_point_detection))
    if s_cfg.get("checkpoint_cleanup", True):
        jobs.append(("checkpoint_cleanup", cleanup_checkpoints))
    if s_cfg.get("trade_stats", True):
        jobs.append(("trade_stats", run_trade_analysis))
    if jobs:
        _schedule_jobs(jobs)
    _started = True
    logger.info("Scheduler started with %d job(s)", len(jobs))

__all__ = [
    "start_scheduler",
    "cleanup_checkpoints",
    "resource_reprobe",
    "run_drift_detection",
    "run_change_point_detection",
    "run_trade_analysis",
]
