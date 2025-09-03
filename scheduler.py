from __future__ import annotations

import asyncio
import logging
import os
import subprocess
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
    s_cfg = cfg.get("scheduler", {}) if isinstance(cfg, dict) else {}
    jobs: list[tuple[str, Callable[[], None | asyncio.Future]]] = []
    if s_cfg.get("resource_reprobe", True):
        jobs.append(("resource_reprobe", resource_reprobe))
    if s_cfg.get("drift_detection", True):
        jobs.append(("drift_detection", run_drift_detection))
    if s_cfg.get("feature_importance_drift", True):
        jobs.append(("feature_importance_drift", run_feature_importance_drift))
    if s_cfg.get("change_point_detection", True):
        jobs.append(("change_point_detection", run_change_point_detection))
    if s_cfg.get("checkpoint_cleanup", True):
        jobs.append(("checkpoint_cleanup", cleanup_checkpoints))
    if s_cfg.get("trade_stats", True):
        jobs.append(("trade_stats", run_trade_analysis))
    if s_cfg.get("decision_review", True):
        jobs.append(("decision_review", run_decision_review))
    if s_cfg.get("vacuum_history", True):
        jobs.append(("vacuum_history", vacuum_history))
    if s_cfg.get("diagnostics", True):
        jobs.append(("diagnostics", run_diagnostics))
    if s_cfg.get("backups", True):
        jobs.append(("backups", run_backups))
    if s_cfg.get("regime_performance", True):
        jobs.append(("regime_performance", update_regime_performance))
    if s_cfg.get("news_vector_store", True):
        jobs.append(("news_vector_store", rebuild_news_vectors))
    if s_cfg.get("world_model_eval", True):
        jobs.append(("world_model_eval", reevaluate_world_model))
    if s_cfg.get("factor_update", True):
        from analysis.factor_updater import update_factors

        jobs.append(("factor_update", update_factors))
    if jobs:
        _schedule_jobs(jobs)
    _started = True
    logger.info("Scheduler started with %d job(s)", len(jobs))

__all__ = [
    "start_scheduler",
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
]
