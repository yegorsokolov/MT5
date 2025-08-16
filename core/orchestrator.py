import asyncio
import logging
import os

from utils.resource_monitor import ResourceMonitor, monitor
import model_registry
import plugins  # noqa: F401 - imported for side effects
import state_manager
from analysis import replay
from . import state_sync
from analytics.metrics_store import record_metric
from risk_manager import risk_manager
from deployment.canary import CanaryManager


class Orchestrator:
    """Central coordinator for resource-aware components."""

    def __init__(self, mon: ResourceMonitor = monitor) -> None:
        self.logger = logging.getLogger(__name__)
        self.monitor = mon
        # Disable automatic refresh; orchestrator controls timing
        self.registry = model_registry.ModelRegistry(monitor=self.monitor, auto_refresh=False)
        self.canary = CanaryManager(self.registry)
        self.checkpoint = None

    async def _watch(self) -> None:
        """React to capability tier upgrades."""
        queue = self.monitor.subscribe()
        previous = self.monitor.capability_tier
        while True:
            tier = await queue.get()
            if model_registry.TIERS.get(tier, 0) > model_registry.TIERS.get(previous, 0):
                self.logger.info("Higher-tier resources detected: %s", tier)
                self.registry.refresh()
                try:
                    reprocess = getattr(replay, "reprocess", None)
                    if callable(reprocess):
                        reprocess()
                    else:
                        getattr(replay, "main", lambda: None)()
                except Exception:  # pragma: no cover - defensive
                    self.logger.exception("Replay reprocess failed")
            previous = tier

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
        self.monitor.start()
        self.registry.refresh()
        self._resume()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        loop.create_task(self._watch())
        loop.create_task(self._sync_monitor())
        loop.create_task(self._daily_summary())

    @classmethod
    def start(cls) -> "Orchestrator":
        orchestrator = cls()
        orchestrator._start()
        return orchestrator

    async def _sync_monitor(self) -> None:
        """Periodically verify that state replication is healthy."""
        interval = int(os.getenv("SYNC_CHECK_INTERVAL", "60"))
        max_lag = int(os.getenv("MAX_SYNC_LAG", "300"))
        while True:
            if not state_sync.check_health(max_lag):
                self.logger.warning("State replication lag exceeds %s seconds", max_lag)
            await asyncio.sleep(interval)

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
            except Exception:
                self.logger.exception("Failed to push daily summary")
            await asyncio.sleep(24 * 60 * 60)
