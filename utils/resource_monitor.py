import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable

import psutil
try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore


@dataclass
class ResourceCapabilities:
    """Basic view of available system resources."""

    cpus: int
    memory_gb: float
    has_gpu: bool
    gpu_count: int

    def model_size(self) -> str:
        """Return "full" for capable machines else "lite"."""
        if self.cpus >= 4 and self.memory_gb >= 16 and self.has_gpu:
            return "full"
        return "lite"

    def ddp(self) -> bool:
        """Return True if DistributedDataParallel should be enabled."""
        return self.gpu_count > 1


class ResourceMonitor:
    """Monitor and periodically refresh system resource info."""

    def __init__(
        self,
        max_rss_mb: Optional[float] = None,
        max_cpu_pct: Optional[float] = None,
        sample_interval: float = 5.0,
        breach_duration: float = 30.0,
        alert_callback: Optional[Callable[[str], Optional[Awaitable[None]]]] = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.capabilities = self._probe()
        self._task: Optional[asyncio.Task] = None
        self._watch_task: Optional[asyncio.Task] = None
        self.max_rss_mb = max_rss_mb
        self.max_cpu_pct = max_cpu_pct
        self.sample_interval = sample_interval
        self._breach_checks = 0
        self._breach_threshold = int(max(breach_duration / sample_interval, 1))
        self.alert_callback = alert_callback

    def _probe(self) -> ResourceCapabilities:
        cpus = psutil.cpu_count(logical=True) or 1
        memory_gb = psutil.virtual_memory().total / (1024**3)
        gpu_count = int(torch.cuda.device_count()) if torch and torch.cuda else 0
        has_gpu = gpu_count > 0
        return ResourceCapabilities(
            cpus=cpus, memory_gb=memory_gb, has_gpu=has_gpu, gpu_count=gpu_count
        )

    async def _periodic_probe(self) -> None:
        while True:
            await asyncio.sleep(24 * 60 * 60)
            self.capabilities = self._probe()
            self.logger.info("Refreshed resource capabilities: %s", self.capabilities)

    async def _watch_usage(self) -> None:
        if not (self.max_rss_mb or self.max_cpu_pct):
            return
        proc = psutil.Process()
        proc.cpu_percent()
        while True:
            await asyncio.sleep(self.sample_interval)
            rss = proc.memory_info().rss / (1024**2)
            cpu = proc.cpu_percent()
            reasons = []
            if self.max_rss_mb and rss > self.max_rss_mb:
                reasons.append(f"rss {rss:.1f}MB>{self.max_rss_mb}")
            if self.max_cpu_pct and cpu > self.max_cpu_pct:
                reasons.append(f"cpu {cpu:.1f}%>{self.max_cpu_pct}%")
            if reasons:
                self._breach_checks += 1
                if self._breach_checks >= self._breach_threshold:
                    msg = ", ".join(reasons)
                    self.logger.error("Resource limits exceeded: %s", msg)
                    if self.alert_callback:
                        try:
                            res = self.alert_callback(msg)
                            if asyncio.iscoroutine(res):
                                await res
                        except Exception:
                            self.logger.exception("Alert callback failed")
                    self._breach_checks = 0
            else:
                self._breach_checks = 0

    def start(self) -> None:
        """Start periodic background probing."""
        if self._task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._periodic_probe())
        if (self.max_rss_mb or self.max_cpu_pct) and self._watch_task is None:
            self._watch_task = loop.create_task(self._watch_usage())

    def stop(self) -> None:
        for attr in ("_task", "_watch_task"):
            task = getattr(self, attr, None)
            if task:
                task.cancel()
                setattr(self, attr, None)


monitor = ResourceMonitor()
