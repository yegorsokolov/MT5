from __future__ import annotations

import asyncio
from concurrent.futures import Future
import atexit
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


try:
    from utils import load_config
except Exception:  # pragma: no cover - config loading optional
    load_config = lambda: {}

logger = logging.getLogger(__name__)

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_started = False
_tasks: list[Future[Any]] = []
_last_retrain_ts: str | None = None
_failed_retrain_attempts: dict[str, int] = {}
_retrain_watcher: Future[Any] | None = None


def _noop() -> None:
    """Placeholder job used when configuration disables a task."""

    return None


def _ensure_background_loop() -> asyncio.AbstractEventLoop:
    """Return the scheduler's background event loop, creating it if needed."""

    global _loop, _thread
    if _loop is None or _thread is None or not _thread.is_alive():
        _loop = asyncio.new_event_loop()
        _thread = threading.Thread(target=_loop.run_forever, daemon=True)
        _thread.start()
    return _loop

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

def _schedule_jobs(
    jobs: Iterable[tuple[str, bool, Callable[[], None | asyncio.Future]]]
) -> int:
    scheduled = 0
    for name, enabled, func in jobs:
        if not enabled:
            logger.info("Skipping disabled scheduler job: %s", name)
            continue
        loop = _ensure_background_loop()
        task = asyncio.run_coroutine_threadsafe(
            _runner(name, 24 * 60 * 60, func), loop
        )
        _tasks.append(task)
        scheduled += 1
        logger.info("Scheduled job: %s", name)
    return scheduled


def stop_scheduler() -> None:
    """Stop the background scheduler and cancel all tasks."""
    global _loop, _thread, _tasks, _started, _retrain_watcher
    if _loop is None:
        _retrain_watcher = None
        _tasks.clear()
        _started = False
        return
    tasks = list(_tasks)
    if _retrain_watcher is not None and _retrain_watcher not in tasks:
        tasks.append(_retrain_watcher)
    for task in tasks:
        task.cancel()
    try:
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), _loop).result(timeout=0.5)
    except Exception:
        pass
    for task in tasks:
        try:
            if not task.done():
                task.result(timeout=0.5)
        except Exception:
            pass
    _tasks.clear()
    _retrain_watcher = None
    try:
        _loop.call_soon_threadsafe(_loop.stop)
    except Exception:
        pass
    if _thread and _thread.is_alive():
        _thread.join()
    try:
        _loop.close()
    except Exception:
        pass
    _loop = None
    _thread = None
    _started = False
    _failed_retrain_attempts.clear()


atexit.register(stop_scheduler)


def _training_cmd(model: str) -> list[str]:
    if model == "nn":
        return ["python", "train_cli.py", "neural", "--resume-online"]
    if model == "rl":
        return ["python", "train_rl.py"]
    return ["python", "train.py", "--resume-online"]


def schedule_retrain(
    model: str = "classic",
    update_hyperparams: bool = False,
    store: "EventStore" | None = None,
) -> None:
    """Record a retrain event for later processing."""
    if store is None:
        from event_store import EventStore

        store = EventStore()
    payload: dict[str, object] = {"model": model}
    if update_hyperparams:
        payload["update_hyperparams"] = True
    store.record("retrain", payload)
    logger.info("Scheduled retrain for %s", model)


def process_retrain_events(store: EventStore | None = None) -> None:
    """Consume retrain events and launch training scripts."""
    try:
        from analytics.metrics_store import log_retrain_outcome
    except Exception:  # pragma: no cover - optional
        from analytics.metrics_aggregator import log_retrain_outcome  # type: ignore
    if store is None:
        from event_store import EventStore

        store = EventStore()

    global _last_retrain_ts, _failed_retrain_attempts
    for ev in store.iter_events("retrain"):
        ts = str(ev["timestamp"])
        payload = ev.get("payload", {})
        model = str(payload.get("model", "classic"))
        attempt_key = f"{ts}|{model}"
        if (
            _last_retrain_ts is not None
            and ts <= _last_retrain_ts
            and attempt_key not in _failed_retrain_attempts
        ):
            continue
        cmd = _training_cmd(model)
        env = os.environ.copy()
        ckpt = payload.get("checkpoint_dir")
        if ckpt:
            env["CHECKPOINT_DIR"] = ckpt
        try:
            subprocess.run(cmd, check=True, env=env)
        except Exception:
            attempts = _failed_retrain_attempts.get(attempt_key, 0) + 1
            _failed_retrain_attempts[attempt_key] = attempts
            logger.exception(
                "Retraining failed for %s", model,
                extra={
                    "model": model,
                    "event_timestamp": ts,
                    "attempt": attempts,
                },
            )
            try:
                log_retrain_outcome(model, "failed")
            except Exception:
                pass
            try:
                log_retrain_outcome(model, "retry_scheduled")
            except Exception:
                pass
            logger.info(
                "Scheduled retrain retry",
                extra={
                    "model": model,
                    "event_timestamp": ts,
                    "attempt": attempts,
                },
            )
            continue
        _failed_retrain_attempts.pop(attempt_key, None)
        try:
            log_retrain_outcome(model, "success")
        except Exception:
            pass
        if _last_retrain_ts is None:
            _last_retrain_ts = ts
        else:
            _last_retrain_ts = max(_last_retrain_ts, ts)


def subscribe_retrain_events(
    store: EventStore | None = None, interval: float = 60.0
) -> Future[Any] | None:
    """Start background task that checks for retrain events."""
    if store is None:
        from event_store import EventStore

        store = EventStore()

    async def _watch() -> None:
        while True:
            process_retrain_events(store)
            await asyncio.sleep(interval)

    global _retrain_watcher
    if _retrain_watcher is not None:
        if not _retrain_watcher.done():
            return _retrain_watcher
        try:
            _tasks.remove(_retrain_watcher)
        except ValueError:
            pass
        _retrain_watcher = None
    loop = _ensure_background_loop()
    future = asyncio.run_coroutine_threadsafe(_watch(), loop)
    _tasks.append(future)
    _retrain_watcher = future
    return future

def cleanup_checkpoints() -> None:
    """Remove old checkpoints, keeping the most recent files."""
    path = Path(os.getenv("CHECKPOINT_DIR", "checkpoints"))
    keep = int(os.getenv("KEEP_CHECKPOINTS", "5"))
    if not path.exists():
        return
    checkpoints = sorted(
        path.glob("checkpoint_*.pkl.enc"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for ckpt in checkpoints[keep:]:
        try:
            ckpt.unlink()
            logger.info("Removed old checkpoint %s", ckpt)
        except Exception:
            logger.exception("Failed removing checkpoint %s", ckpt)

async def resource_reprobe() -> None:
    """Refresh resource capability information."""
    try:
        from utils.resource_monitor import monitor

        await monitor.probe()
    except Exception:
        logger.exception("Resource reprobe failed")
    try:
        import pandas as pd
        import numpy as np
        from analysis.domain_adapter import DomainAdapter
        from monitor_drift import DRIFT_METRICS

        adapter = DomainAdapter.load(Path("domain_adapter.pkl"))
        if DRIFT_METRICS.exists():
            df = pd.read_parquet(DRIFT_METRICS)
            num = df.select_dtypes(np.number)
            if not num.empty:
                adapter.reestimate(num)
                adapter.save(Path("domain_adapter.pkl"))
        logger.info("Domain adapter parameters refreshed")
    except Exception:
        logger.exception("Domain adapter re-estimation failed")
    try:
        from signal_queue import _ROUTER

        _ROUTER.refresh_regime_performance()
        logger.info("Strategy router regime performance refreshed")
    except Exception:
        logger.exception("Strategy router refresh failed")


def rebuild_news_vectors() -> None:
    """Rebuild the news event vector store with latest cached events."""

    try:
        from news.aggregator import NewsAggregator
        from news import vector_store

        agg = NewsAggregator()
        events = agg.fetch()
        texts = [ev.get("event") for ev in events if ev.get("event")]
        vector_store.rebuild(texts)
        logger.info("Rebuilt news vector store with %d events", len(texts))
    except Exception:
        logger.exception("News vector store rebuild failed")

def run_drift_detection() -> None:
    """Run model/data drift comparison."""
    try:
        from monitor_drift import monitor

        monitor.compare()
        logger.info("Drift detection completed")
    except Exception:
        logger.exception("Drift detection failed")


def run_feature_importance_drift() -> None:
    """Analyze feature importance drift and persist reports."""
    try:
        from analysis.feature_importance_drift import analyze

        flagged = analyze()
        if flagged:
            logger.warning("Feature importance drift detected: %s", flagged)
        else:
            logger.info("No feature importance drift detected")
    except Exception:
        logger.exception("Feature importance drift analysis failed")


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


def update_regime_performance() -> None:
    """Recompute regime/model performance statistics and refresh router."""
    try:
        from analytics.regime_performance_store import RegimePerformanceStore

        store = RegimePerformanceStore()
        store.recompute()
        from signal_queue import _ROUTER

        _ROUTER.refresh_regime_performance()
        logger.info("Regime performance statistics updated")
    except Exception:
        logger.exception("Regime performance update failed")


def reevaluate_world_model() -> None:
    """Retrain a small world model on recent experiences and log the error.

    The function is intentionally defensive: all imports are done lazily and any
    exception simply results in the routine logging the failure.  When
    successful, the mean squared error of the reward predictions is written to
    the :mod:`model_store` for later inspection.
    """

    try:
        from rl.world_model import WorldModel, Transition  # type: ignore
        from rl.offline_dataset import OfflineDataset  # type: ignore
        import numpy as np  # type: ignore
        from models import model_store  # type: ignore
    except Exception:  # pragma: no cover - optional dependencies missing
        logger.exception("World model evaluation dependencies unavailable")
        return

    try:
        dataset = OfflineDataset()
        if not dataset.samples:
            dataset.close()
            return
        first = dataset.samples[0]
        wm = WorldModel(len(first.obs), len(first.action))
        transitions = [
            Transition(s.obs, s.action, s.next_obs, s.reward) for s in dataset.samples
        ]
        wm.train(transitions)
        preds = [wm.predict(t.state, t.action)[1] for t in transitions]
        rewards = [t.reward for t in transitions]
        mse = float(np.mean((np.array(preds) - np.array(rewards)) ** 2))
        model_store.save_replay_stats({"world_model_mse": mse})
        dataset.close()
        logger.info("World model re-evaluation MSE %.6f", mse)
    except Exception:
        logger.exception("World model re-evaluation failed")


def run_decision_review() -> None:
    """Run LLM-based review of recorded decision rationales."""
    try:
        from analysis.decision_reviewer import review_rationales

        flagged = review_rationales()
        if flagged.get("manual"):
            logger.warning("Decisions flagged for manual follow-up: %s", flagged["manual"])
        if flagged.get("retrain"):
            logger.warning("Decisions flagged for retraining: %s", flagged["retrain"])
    except Exception:
        logger.exception("Decision review failed")


def run_diagnostics() -> None:
    """Run system diagnostics and dependency checks."""
    try:
        subprocess.run(["bash", "scripts/diagnostics.sh"], check=True)
        logger.info("Diagnostics completed")
    except Exception:
        logger.exception("Diagnostics failed")


def vacuum_history(path: Path | None = None) -> None:
    """Compact partitioned Parquet datasets by merging small files."""
    try:
        import pyarrow.dataset as ds  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        logger.exception("Vacuum dependencies unavailable")
        return

    base = Path(path) if path else Path(os.getenv("HISTORY_DATASET", "data/history.parquet"))
    if not base.exists():
        return

    for part in base.rglob("*"):
        if not part.is_dir():
            continue
        files = list(part.glob("*.parquet"))
        if len(files) <= 1:
            continue
        try:
            dataset = ds.dataset(part, format="parquet")
            table = dataset.to_table()
            tmp = part / "_tmp.parquet"
            pq.write_table(table, tmp, compression="zstd")
            for f in files:
                f.unlink(missing_ok=True)  # type: ignore[attr-defined]
            tmp.rename(part / "part-0.parquet")
        except Exception:
            logger.exception("Vacuum failed for %s", part)

    # prune empty directories
    for d in sorted(base.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if d.is_dir() and not list(d.glob("*.parquet")) and not any(d.iterdir()):
            try:
                d.rmdir()
            except Exception:
                logger.exception("Failed removing empty partition %s", d)


def run_backups() -> None:
    """Invoke the backup manager to archive checkpoints and logs."""
    try:
        from core.backup_manager import BackupManager

        BackupManager().run()
        logger.info("Backup manager completed")
    except Exception:
        logger.exception("Backup manager failed")

def start_scheduler() -> None:
    """Start background scheduler based on configuration."""
    global _started
    if _started:
        return
    cfg = load_config()
    scheduler_section: Mapping[str, Any] | None = None
    if hasattr(cfg, "get"):
        scheduler_section = cfg.get("scheduler")
        if scheduler_section is None:
            scheduler_section = cfg.get("scheduler", {})
    if scheduler_section is None and hasattr(cfg, "model_dump"):
        try:
            scheduler_section = cfg.model_dump().get("scheduler")
        except Exception:  # pragma: no cover - defensive
            scheduler_section = None
    if scheduler_section is None:
        s_cfg: dict[str, Any] = {}
    elif isinstance(scheduler_section, Mapping):
        s_cfg = dict(scheduler_section)
    else:
        try:
            s_cfg = dict(scheduler_section)
        except TypeError:
            s_cfg = {}

    def _flag(name: str, default: bool = True) -> bool:
        value = s_cfg.get(name, default)
        return bool(value) if value is not None else False

    if _flag("retrain_events", True):
        try:
            subscribe_retrain_events()
        except Exception:
            logger.exception("Retrain event subscription failed")
    else:
        logger.info("Skipping disabled scheduler job: retrain_events")
    jobs_with_flags: list[tuple[str, bool, Callable[[], None | asyncio.Future]]] = [
        ("resource_reprobe", _flag("resource_reprobe", True), resource_reprobe),
        ("drift_detection", _flag("drift_detection", True), run_drift_detection),
        (
            "feature_importance_drift",
            _flag("feature_importance_drift", True),
            run_feature_importance_drift,
        ),
        (
            "change_point_detection",
            _flag("change_point_detection", True),
            run_change_point_detection,
        ),
        ("checkpoint_cleanup", _flag("checkpoint_cleanup", True), cleanup_checkpoints),
        ("trade_stats", _flag("trade_stats", True), run_trade_analysis),
        ("decision_review", _flag("decision_review", True), run_decision_review),
        ("vacuum_history", _flag("vacuum_history", True), vacuum_history),
        ("diagnostics", _flag("diagnostics", True), run_diagnostics),
        ("backups", _flag("backups", True), run_backups),
        (
            "regime_performance",
            _flag("regime_performance", True),
            update_regime_performance,
        ),
        (
            "news_vector_store",
            _flag("news_vector_store", True),
            rebuild_news_vectors,
        ),
        ("world_model_eval", _flag("world_model_eval", True), reevaluate_world_model),
    ]
    factor_enabled = _flag("factor_update", True)
    if factor_enabled:
        from analysis.factor_updater import update_factors

        factor_func = update_factors
    else:
        factor_func = _noop
    jobs_with_flags.append(("factor_update", factor_enabled, factor_func))
    scheduled_jobs = _schedule_jobs(jobs_with_flags)
    _started = True
    logger.info("Scheduler started with %d job(s)", scheduled_jobs)

__all__ = [
    "start_scheduler",
    "stop_scheduler",
    "cleanup_checkpoints",
    "resource_reprobe",
    "run_drift_detection",
    "run_feature_importance_drift",
    "run_change_point_detection",
    "run_trade_analysis",
    "run_decision_review",
    "run_diagnostics",
    "vacuum_history",
    "run_backups",
    "update_regime_performance",
    "rebuild_news_vectors",
    "reevaluate_world_model",
    "schedule_retrain",
    "process_retrain_events",
    "subscribe_retrain_events",
]
