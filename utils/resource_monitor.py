import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable, Set, Dict, Deque

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
        self._usage_subscribers: Set[asyncio.Queue[Dict[str, float]]] = set()
        self.max_rss_mb = max_rss_mb
        self.max_cpu_pct = max_cpu_pct
        self.sample_interval = sample_interval
        self._breach_checks = 0
        self._breach_threshold = int(max(breach_duration / sample_interval, 1))
        self.alert_callback = alert_callback
        self.latest_usage: Dict[str, object] = {}
        self._tick_start: Optional[float] = None
        self.tick_to_signal_latency: float = 0.0
        self._module_procs: Dict[str, psutil.Process] = {}
        self._module_samples: Dict[str, Dict[str, Deque[float]]] = {}
        self._module_averages: Dict[str, Dict[str, float]] = {}
        self._module_window = 12

    def register_module(self, name: str, pid: Optional[int] = None) -> None:
        """Track CPU usage for an additional module/process.

        Parameters
        ----------
        name:
            Identifier for the module being monitored.
        pid:
            Optional process ID.  Defaults to the current process.
        """

        if name in self._module_procs:
            return
        try:
            proc = psutil.Process(pid) if pid is not None else psutil.Process()
            proc.cpu_percent()
        except Exception:
            return
        self._module_procs[name] = proc
        self._module_samples[name] = {
            "cpu": deque(maxlen=self._module_window),
            "power": deque(maxlen=self._module_window),
        }
        self._module_averages[name] = {"cpu_avg": 0.0, "power_avg": 0.0}

    def module_averages(self) -> Dict[str, Dict[str, float]]:
        """Return rolling average metrics for registered modules."""

        return self._module_averages

    def mark_tick(self) -> None:
        """Record the arrival time of the latest tick."""
        self._tick_start = time.perf_counter()

    def mark_signal(self) -> None:
        """Record signal emission and update latency metrics."""
        if self._tick_start is None:
            return
        latency = time.perf_counter() - self._tick_start
        self.tick_to_signal_latency = latency
        try:
            record_metric("tick_to_signal_latency", latency)
        except Exception:
            pass
        self._tick_start = None

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

    async def probe(self) -> None:
        """Refresh capabilities and broadcast any changes to subscribers."""

        old_tier = self.capability_tier
        self.capabilities = self._probe()
        new_tier = self.capabilities.capability_tier()
        self.capability_tier = new_tier
        # Notify subscribers even if the tier did not change so they can
        # reconsider heavier models when hardware improves within the same tier.
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
        try:
            from model_registry import TIERS, select_models
            from analysis import replay

            if TIERS.get(new_tier, 0) > TIERS.get(old_tier, 0):
                select_models()
                await asyncio.to_thread(replay.reprocess_trades)
        except Exception:
            self.logger.debug("Model reload or replay failed", exc_info=True)

    async def _periodic_probe(self) -> None:
        while True:
            await asyncio.sleep(24 * 60 * 60)
            await self.probe()

    async def _watch_usage(self) -> None:
        proc = psutil.Process()
        proc.cpu_percent()
        for mod_proc in self._module_procs.values():
            try:
                mod_proc.cpu_percent()
            except Exception:
                pass
        prev_net = self._read_net_bytes()
        prev_disk = psutil.disk_io_counters()
        while True:
            await asyncio.sleep(self.sample_interval)
            rss = proc.memory_info().rss / (1024**2)
            cpu = proc.cpu_percent()
            power = cpu
            recv, sent = self._read_net_bytes()
            disk = psutil.disk_io_counters()
            net_rx = (recv - prev_net[0]) / self.sample_interval
            net_tx = (sent - prev_net[1]) / self.sample_interval
            disk_read = (disk.read_bytes - prev_disk.read_bytes) / self.sample_interval
            disk_write = (
                disk.write_bytes - prev_disk.write_bytes
            ) / self.sample_interval
            prev_net = (recv, sent)
            prev_disk = disk
            try:
                record_metric("rss_usage_mb", rss)
                record_metric("cpu_usage_pct", cpu)
                record_metric("power_proxy", power)
                record_metric("net_rx_bytes", net_rx)
                record_metric("net_tx_bytes", net_tx)
                record_metric("disk_read_bytes", disk_read)
                record_metric("disk_write_bytes", disk_write)
            except Exception:
                pass
            usage = {
                "rss_mb": rss,
                "cpu_pct": cpu,
                "power_proxy": power,
                "net_rx": net_rx,
                "net_tx": net_tx,
                "disk_read": disk_read,
                "disk_write": disk_write,
            }
            modules: Dict[str, Dict[str, float]] = {}
            for name, mproc in list(self._module_procs.items()):
                try:
                    m_cpu = mproc.cpu_percent()
                except Exception:
                    continue
                m_power = m_cpu
                samples = self._module_samples.get(name)
                if samples is None:
                    continue
                samples["cpu"].append(m_cpu)
                samples["power"].append(m_power)
                cpu_avg = sum(samples["cpu"]) / len(samples["cpu"])
                power_avg = sum(samples["power"]) / len(samples["power"])
                self._module_averages[name] = {
                    "cpu_avg": cpu_avg,
                    "power_avg": power_avg,
                }
                modules[name] = {
                    "cpu_pct": m_cpu,
                    "power_proxy": m_power,
                    "cpu_avg": cpu_avg,
                    "power_avg": power_avg,
                }
                try:
                    record_metric(f"module_cpu_pct_{name}", m_cpu)
                    record_metric(f"module_power_proxy_{name}", m_power)
                except Exception:
                    pass
            if modules:
                usage["modules"] = modules
            self.latest_usage = usage
            await self._notify_usage(usage)
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

    def subscribe_usage(self) -> asyncio.Queue[Dict[str, float]]:
        """Return a queue that receives periodic resource usage samples."""
        q: asyncio.Queue[Dict[str, float]] = asyncio.Queue()
        self._usage_subscribers.add(q)
        return q

    async def _notify(self, tier: str) -> None:
        for q in list(self._subscribers):
            try:
                await q.put(tier)
            except Exception:
                self._subscribers.discard(q)

    async def _notify_usage(self, usage: Dict[str, float]) -> None:
        for q in list(self._usage_subscribers):
            try:
                await q.put(usage)
            except Exception:
                self._usage_subscribers.discard(q)

    def stop(self) -> None:
        for attr in ("_task", "_watch_task"):
            task = getattr(self, attr, None)
            if task:
                task.cancel()
                setattr(self, attr, None)

    def _read_net_bytes(self) -> tuple[int, int]:
        """Return total received and sent bytes across all interfaces."""
        recv = sent = 0
        try:
            with open("/proc/net/dev", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if ":" not in line:
                        continue
                    _, data = line.split(":", 1)
                    fields = data.split()
                    if len(fields) >= 9:
                        recv += int(fields[0])
                        sent += int(fields[8])
        except Exception:
            pass
        return recv, sent


monitor = ResourceMonitor()


def main() -> None:
    """Entry point used by the Docker image.

    It simply loads any registered plugins which can enable or disable
    functionality based on the detected :data:`capability_tier` and prints the
    tier for visibility.  This keeps the Docker entrypoint lightweight while
    still allowing the monitor to be invoked as a standalone module.
    """

    try:
        from plugins import PLUGIN_SPECS

        for spec in PLUGIN_SPECS:
            spec.load()
    except Exception:  # pragma: no cover - plugins are optional
        pass
    print(f"Resource tier detected: {monitor.capability_tier}")


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
