import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable, Set

import psutil
try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore

from analytics.metrics_store import record_metric


@dataclass
class ResourceCapabilities:
    """Basic view of available system resources."""

    cpus: int
    memory_gb: float
    has_gpu: bool
    gpu_count: int
    gpu_model: str = ""
    cpu_flags: Set[str] = field(default_factory=set)

    def capability_tier(self) -> str:
        """Classify the machine into capability tiers.

        ``lite`` machines have limited resources. ``standard`` machines have
        reasonable CPU and memory but no GPU. ``gpu`` machines have at least
        one GPU, while ``hpc`` represents multi-GPU or data-centre class
        hardware (A100/V100/H100 etc. or very high CPU/RAM).
        """

        gpu_name = self.gpu_model.lower()
        if self.has_gpu:
            if (
                self.gpu_count > 1
                or any(t in gpu_name for t in ["a100", "h100", "v100", "a40", "a30"])
                or (self.cpus >= 16 and self.memory_gb >= 64)
            ):
                return "hpc"
            return "gpu"
        if self.cpus >= 4 and self.memory_gb >= 16:
            return "standard"
        return "lite"

    # Backwards compatibility
    def model_size(self) -> str:  # pragma: no cover - legacy alias
        return self.capability_tier()

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
        self.capability_tier = self.capabilities.capability_tier()
        self._enable_accelerated_libraries()
        self._task: Optional[asyncio.Task] = None
        self._watch_task: Optional[asyncio.Task] = None
        self._subscribers: Set[asyncio.Queue[str]] = set()
        self.max_rss_mb = max_rss_mb
        self.max_cpu_pct = max_cpu_pct
        self.sample_interval = sample_interval
        self._breach_checks = 0
        self._breach_threshold = int(max(breach_duration / sample_interval, 1))
        self.alert_callback = alert_callback

    def _parse_cpu_flags(self) -> Set[str]:
        flags: Set[str] = set()
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.lower().startswith("flags"):
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            flags.update(parts[1].strip().split())
                        break
        except OSError:
            self.logger.debug("/proc/cpuinfo not available for flag detection")
        return flags

    def _probe(self) -> ResourceCapabilities:
        cpus = psutil.cpu_count(logical=True) or 1
        memory_gb = psutil.virtual_memory().total / (1024**3)
        gpu_count = int(torch.cuda.device_count()) if torch and torch.cuda else 0
        has_gpu = gpu_count > 0
        gpu_model = ""
        if has_gpu and torch and torch.cuda:
            try:
                gpu_model = torch.cuda.get_device_name(0)
            except Exception:
                gpu_model = ""
        flags = self._parse_cpu_flags()
        return ResourceCapabilities(
            cpus=cpus,
            memory_gb=memory_gb,
            has_gpu=has_gpu,
            gpu_count=gpu_count,
            gpu_model=gpu_model,
            cpu_flags=flags,
        )

    def _enable_accelerated_libraries(self) -> None:
        flags = self.capabilities.cpu_flags
        if not ({"avx2", "fma"} & flags):
            self.logger.info("AVX2/FMA not detected; using default libraries")
            return
        try:  # pragma: no cover - optional dependency
            import numexpr

            numexpr.set_num_threads(self.capabilities.cpus)
            self.logger.info("Enabled numexpr with %s threads", self.capabilities.cpus)
        except Exception:
            self.logger.debug("numexpr unavailable; skipping acceleration")
        try:
            import pandas as pd

            try:
                import pyarrow  # noqa: F401

                pd.options.mode.dtype_backend = "pyarrow"
                self.logger.info("Enabled pandas PyArrow backend")
            except Exception:
                self.logger.debug("pyarrow unavailable; using default pandas backend")
        except Exception:
            self.logger.debug("pandas not installed; cannot set dtype backend")

    async def _periodic_probe(self) -> None:
        while True:
            await asyncio.sleep(24 * 60 * 60)
            self.capabilities = self._probe()
            new_tier = self.capabilities.capability_tier()
            if new_tier != self.capability_tier:
                self.capability_tier = new_tier
                await self._notify(new_tier)
            self.logger.info(
                "Refreshed resource capabilities: %s (tier=%s)",
                self.capabilities,
                self.capability_tier,
            )
            try:
                from risk_manager import risk_manager

                risk_manager.rebalance_budgets()
            except Exception:
                self.logger.debug("Risk budget rebalance failed", exc_info=True)

    async def _watch_usage(self) -> None:
        proc = psutil.Process()
        proc.cpu_percent()
        while True:
            await asyncio.sleep(self.sample_interval)
            rss = proc.memory_info().rss / (1024**2)
            cpu = proc.cpu_percent()
            try:
                record_metric("rss_usage_mb", rss)
                record_metric("cpu_usage_pct", cpu)
            except Exception:
                pass
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
        if self._watch_task is None:
            self._watch_task = loop.create_task(self._watch_usage())

    def subscribe(self) -> asyncio.Queue[str]:
        """Return a queue that receives capability tier updates."""
        q: asyncio.Queue[str] = asyncio.Queue()
        self._subscribers.add(q)
        return q

    async def _notify(self, tier: str) -> None:
        for q in list(self._subscribers):
            try:
                await q.put(tier)
            except Exception:
                self._subscribers.discard(q)

    def stop(self) -> None:
        for attr in ("_task", "_watch_task"):
            task = getattr(self, attr, None)
            if task:
                task.cancel()
                setattr(self, attr, None)


monitor = ResourceMonitor()
