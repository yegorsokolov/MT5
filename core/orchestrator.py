import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

try:  # pragma: no cover - alerting optional in tests
    from utils.alerting import send_alert
except Exception:  # pragma: no cover - fallback stub

    def send_alert(msg: str) -> None:  # type: ignore
        return


from utils import load_config
from utils.resource_monitor import ResourceMonitor, monitor
from features import start_capability_watch
from mt5 import model_registry
import plugins  # noqa: F401 - imported for side effects
from mt5 import state_manager
from analysis import replay
from . import state_sync
from analytics.metrics_store import record_metric
from mt5 import risk_manager as risk_manager_module
from mt5.risk_manager import risk_manager, subscribe_to_broker_alerts
from deployment.canary import CanaryManager
from news.aggregator import NewsAggregator
from strategy.shadow_runner import ShadowRunner
from strategy.evolution_lab import EvolutionLab


DEFAULT_SERVICE_CMDS: dict[str, list[str]] = {
    "signal_queue": ["python", "-m", "mt5.signal_queue"],
    "realtime_train": ["python", "-m", "mt5.realtime_train"],
}


def _compute_quiet_windows(agg: NewsAggregator, minutes: int) -> list[dict]:
    """Return quiet trading windows around high-impact events."""
    now = datetime.now(timezone.utc)
    try:
        agg.fetch()
    except Exception:
        pass
    events = agg.get_news(now, now + timedelta(days=1))
    windows: list[dict] = []
    for ev in events:
        imp = str(ev.get("importance", "")).lower()
        ts = ev.get("timestamp")
        if ts and (imp.startswith("high") or imp.startswith("red")):
            start = ts - timedelta(minutes=minutes)
            end = ts + timedelta(minutes=minutes)
            currencies = ev.get("currencies")
            if not currencies:
                cur = ev.get("currency")
                currencies = [cur] if cur else []
            symbols = ev.get("symbols", []) or []
            windows.append(
                {
                    "start": start,
                    "end": end,
                    "currencies": currencies,
                    "symbols": symbols,
                }
            )
    return windows


class Orchestrator:
    """Central coordinator for resource-aware components."""

    def __init__(self, mon: ResourceMonitor = monitor) -> None:
        self.logger = logging.getLogger(__name__)
        self.monitor = mon
        # Disable automatic refresh; orchestrator controls timing
        self.registry = model_registry.ModelRegistry(
            monitor=self.monitor, auto_refresh=False
        )
        self.canary = CanaryManager(self.registry)
        self.checkpoint = None
        # mapping of service -> command used to (re)start it
        env_cmds = os.getenv("SERVICE_COMMANDS")
        cfg_cmds: dict[str, list[str]] = {}
        if env_cmds:
            try:
                cfg_cmds = json.loads(env_cmds)
            except Exception:
                self.logger.exception("Invalid SERVICE_COMMANDS; using defaults")
        else:
            try:
                cfg_cmds = load_config().get("service_cmds") or {}
            except Exception:
                cfg_cmds = {}
        self._service_cmds = {**DEFAULT_SERVICE_CMDS, **cfg_cmds}
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._shadow_tasks: dict[str, asyncio.Task] = {}
        self.lab = EvolutionLab(self._base_strategy(), register=self.register_strategy)

    def _base_strategy(self) -> Callable[[dict], float]:
        """Return a default strategy used as seed for variant generation."""

        try:
            from mt5 import signal_queue  # local import to avoid heavy dependency

            router = getattr(signal_queue, "_ROUTER", None)
            algorithms = getattr(router, "algorithms", {}) if router else {}
            if algorithms:
                return next(iter(algorithms.values()))
        except Exception:
            pass
        return lambda _: 0.0

    async def _watch(self) -> None:
        """React to capability tier upgrades."""
        queue = self.monitor.subscribe()
        previous = self.monitor.capability_tier
        while True:
            tier = await queue.get()
            if model_registry.TIERS.get(tier, 0) > model_registry.TIERS.get(
                previous, 0
            ):
                self.logger.info("Higher-tier resources detected: %s", tier)
                self.registry.refresh()
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:  # pragma: no cover - event loop not running
                    loop = asyncio.get_event_loop()
                loop.create_task(self._run_reprocess())
            previous = tier

    async def _run_reprocess(self) -> None:
        """Reprocess trades and alert operators on completion."""
        try:
            df = await asyncio.to_thread(replay.reprocess, Path("reports/replays"))
            diff = float(df["pnl_new"].iloc[0] - df["pnl_old"].iloc[0])
            send_alert(f"Replay reprocess complete. ΔPnL={diff:.2f}")
        except Exception:  # pragma: no cover - defensive
            self.logger.exception("Replay reprocess failed")
            try:
                send_alert("Replay reprocess failed")
            except Exception:
                pass

    def _resume(self) -> None:
        loader = getattr(state_manager, "latest_checkpoint", None)
        if callable(loader):
            self.checkpoint = loader()
        else:
            loader = getattr(state_manager, "load_latest_checkpoint", None)
            if callable(loader):
                self.checkpoint = loader()

    def _start(self) -> None:
        max_rss = float(os.getenv("MAX_RSS_MB", "0") or 0) or None
        max_cpu = float(os.getenv("MAX_CPU_PCT", "0") or 0) or None
        if max_rss is not None:
            self.monitor.max_rss_mb = max_rss
        if max_cpu is not None:
            self.monitor.max_cpu_pct = max_cpu
        starter = getattr(risk_manager_module, "ensure_scheduler_started", None)
        started = False
        if callable(starter):
            try:
                starter()
                started = True
            except Exception:
                self.logger.exception(
                    "Risk manager scheduler helper failed; attempting direct start"
                )
        if not started:
            try:
                from mt5 import scheduler

                scheduler.start_scheduler()
            except Exception:
                self.logger.exception("Scheduler start failed")
        self.monitor.start()
        start_capability_watch()
        self.registry.refresh()
        state_sync.pull_event_store()
        self._resume()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        subscribe_to_broker_alerts()
        loop.create_task(self._watch())
        loop.create_task(self._sync_monitor())
        loop.create_task(self._daily_summary())
        loop.create_task(self._watch_services())
        loop.create_task(self._update_quiet_windows())
        try:  # start shadow runners for existing strategies
            from mt5 import signal_queue

            for name, algo in getattr(signal_queue, "_ROUTER").algorithms.items():
                self.register_strategy(name, algo)
        except Exception:
            pass

    @classmethod
    def start(cls) -> "Orchestrator":
        orchestrator = cls()
        orchestrator._start()
        global GLOBAL_ORCHESTRATOR
        GLOBAL_ORCHESTRATOR = orchestrator
        return orchestrator

    async def _sync_monitor(self) -> None:
        """Periodically verify that state replication is healthy."""
        interval = int(os.getenv("SYNC_CHECK_INTERVAL", "60"))
        max_lag = int(os.getenv("MAX_SYNC_LAG", "300"))
        while True:
            if not state_sync.check_health(max_lag):
                self.logger.warning("State replication lag exceeds %s seconds", max_lag)
            await asyncio.sleep(interval)

    # Shadow strategy management ---------------------------------------------
    def register_strategy(self, name: str, handler: Callable[[dict], float]) -> None:
        """Start a shadow runner for ``name`` if not already running."""
        if name in self._shadow_tasks:
            return
        runner = ShadowRunner(name=name, handler=handler)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        self._shadow_tasks[name] = loop.create_task(runner.run())

    async def _daily_summary(self) -> None:
        """Record daily aggregated metrics to the store."""
        while True:
            try:
                status = risk_manager.status()
                for key, val in status.items():
                    if isinstance(val, (int, float)):
                        record_metric(key, val, {"summary": "daily"})
                # Evaluate any active canary deployments daily
                self.canary.evaluate_all()
                try:
                    from analysis.strategy_evaluator import StrategyEvaluator

                    StrategyEvaluator().run()
                except Exception:
                    self.logger.exception("Strategy evaluation failed")
                try:
                    self.lab.run()
                except Exception:
                    self.logger.exception("Evolution lab failed")
            except Exception:
                self.logger.exception("Failed to push daily summary")
            await asyncio.sleep(24 * 60 * 60)

    async def _update_quiet_windows(self) -> None:
        """Refresh quiet trading windows around high-impact news events."""
        agg = NewsAggregator()
        minutes = int(os.getenv("NEWS_BLACKOUT_MINUTES", "30"))
        while True:
            try:
                windows = _compute_quiet_windows(agg, minutes)
                if hasattr(risk_manager, "set_quiet_windows"):
                    risk_manager.set_quiet_windows(windows)
                record_metric("quiet_windows", len(windows))
            except Exception:
                self.logger.exception("Failed to update quiet windows")
            await asyncio.sleep(24 * 60 * 60)

    async def _watch_services(self) -> None:
        """Monitor critical background services and restart if needed."""

        if os.getenv("SERVICE_WATCHDOG", "0") != "1":
            return

        interval = int(os.getenv("SERVICE_WATCHDOG_INTERVAL", "60"))

        while True:
            # Ensure scheduler thread is alive
            try:
                from mt5 import scheduler  # local import to avoid heavy dependency at import time

                if not getattr(scheduler, "_started", False):
                    scheduler.start_scheduler()
            except Exception:
                self.logger.exception("Scheduler health check failed")

            # Check external service processes
            for name, cmd in self._service_cmds.items():
                proc = self._processes.get(name)
                if proc is None or proc.poll() is not None:
                    self.logger.warning("Service %s not running; restarting", name)
                    try:
                        self._processes[name] = subprocess.Popen(cmd)
                        count = state_manager.increment_restart(name)
                        record_metric(f"{name}_restarts", count, {"service": name})
                        send_alert(f"Service {name} restarted")
                    except Exception:
                        self.logger.exception("Failed to restart service %s", name)
            await asyncio.sleep(interval)


GLOBAL_ORCHESTRATOR: Orchestrator | None = None
