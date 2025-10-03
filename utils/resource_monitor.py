import asyncio
import logging
import threading
import time
from collections import deque
from concurrent.futures import Future as ThreadFuture
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable, Set, Dict, Deque, Union, Any

import psutil

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore

from analytics.metrics_store import record_metric

try:  # Allow direct execution without package context in tests
    from . import load_config
except ImportError:  # pragma: no cover - fallback for loose imports
    from utils import load_config  # type: ignore


def _plugin_cache_ttl() -> float:
    """Return plugin cache TTL from configuration."""

    try:
        return float(load_config().get("plugin_cache_ttl", 0) or 0)
    except Exception:
        return 0.0


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


FutureLike = Union[asyncio.Task, ThreadFuture]


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
        self._task: Optional[FutureLike] = None
        self._watch_task: Optional[FutureLike] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._owns_loop = False
        self._tasks: Dict[FutureLike, Optional[asyncio.AbstractEventLoop]] = {}
        self._tasks_lock = threading.Lock()
        self._subscribers: Set[
            tuple[asyncio.Queue[str], asyncio.AbstractEventLoop]
        ] = set()
        self._usage_subscribers: Set[
            tuple[asyncio.Queue[Dict[str, float]], asyncio.AbstractEventLoop]
        ] = set()
        self.max_rss_mb = max_rss_mb
        self.max_cpu_pct = max_cpu_pct
        self.sample_interval = sample_interval
        self._breach_checks = 0
        self._breach_threshold = int(max(breach_duration / sample_interval, 1))
        self.alert_callback = alert_callback
        self.latest_usage: Dict[str, object] = {}
        self._tick_start: Optional[float] = None
        self.tick_to_signal_latency: float = 0.0
        self._latency_samples: Deque[float] = deque(maxlen=120)
        self.avg_latency: float = 0.0
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
        self._latency_samples.append(latency)
        self.avg_latency = sum(self._latency_samples) / len(self._latency_samples)
        try:
            record_metric("tick_to_signal_latency", latency)
            record_metric("tick_to_signal_latency_avg", self.avg_latency)
        except Exception:
            pass
        self._tick_start = None

    def latency(self) -> float:
        """Return average tick-to-signal latency."""
        if self._latency_samples:
            return self.avg_latency
        return self.tick_to_signal_latency

    def latency_exceeded(self, threshold: float) -> bool:
        """Return True if average latency is above ``threshold``."""
        return threshold > 0 and self.latency() > threshold

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
            from mt5.controller_settings import auto_tune_controller_settings

            auto_tune_controller_settings(self)
        except Exception:
            self.logger.debug("Controller auto-tune failed", exc_info=True)
        ttl = _plugin_cache_ttl()
        if ttl > 0:
            try:
                import importlib

                plugins = importlib.import_module("plugins")
                plugins.purge_unused_plugins(ttl)
            except Exception:
                self.logger.debug("Plugin purge failed", exc_info=True)
        try:
            from mt5.risk_manager import risk_manager

            risk_manager.rebalance_budgets()
        except Exception:
            self.logger.debug("Risk budget rebalance failed", exc_info=True)
        try:
            from mt5.model_registry import TIERS, select_models
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

    def _future_active(self, fut: Optional[FutureLike]) -> bool:
        return fut is not None and not fut.done()

    def start(self) -> None:
        """Start periodic background probing."""
        if self._task is not None and self._task.done():
            self._task = None
        if self._watch_task is not None and self._watch_task.done():
            self._watch_task = None
        if self._future_active(self._task) and self._future_active(self._watch_task):
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            self._loop = loop
            self._owns_loop = False
            self._loop_thread = None
            if not self._future_active(self._task):
                self._task = loop.create_task(self._periodic_probe())
            if not self._future_active(self._watch_task):
                self._watch_task = loop.create_task(self._watch_usage())
            return

        needs_loop = (
            self._loop is None
            or self._loop.is_closed()
            or not self._owns_loop
            or self._loop_thread is None
            or not self._loop_thread.is_alive()
        )
        if needs_loop:
            self._loop = asyncio.new_event_loop()
            self._owns_loop = True

            def _run(loop_obj: asyncio.AbstractEventLoop) -> None:
                asyncio.set_event_loop(loop_obj)
                loop_obj.run_forever()

            self._loop_thread = threading.Thread(
                target=_run,
                args=(self._loop,),
                daemon=True,
                name="ResourceMonitorLoop",
            )
            self._loop_thread.start()

        assert self._loop is not None
        if not self._future_active(self._task):
            self._task = asyncio.run_coroutine_threadsafe(
                self._periodic_probe(), self._loop
            )
        if not self._future_active(self._watch_task):
            self._watch_task = asyncio.run_coroutine_threadsafe(
                self._watch_usage(), self._loop
            )

    def create_task(self, coro: Awaitable[Any]) -> FutureLike:
        """Schedule ``coro`` on the monitor's event loop."""

        self.start()
        loop = self._loop
        if loop is None:
            raise RuntimeError("Resource monitor loop unavailable")
        if not self._owns_loop:
            return self._track_task(loop.create_task(coro), loop)
        if threading.current_thread() is self._loop_thread:
            return self._track_task(loop.create_task(coro), loop)
        return self._track_task(asyncio.run_coroutine_threadsafe(coro, loop), loop)

    def _track_task(
        self,
        future: FutureLike,
        loop: Optional[asyncio.AbstractEventLoop],
    ) -> FutureLike:
        with self._tasks_lock:
            self._tasks[future] = loop

        def _cleanup(completed: FutureLike) -> None:
            with self._tasks_lock:
                self._tasks.pop(completed, None)

        try:
            future.add_done_callback(_cleanup)  # type: ignore[attr-defined]
        except Exception:
            with self._tasks_lock:
                self._tasks.pop(future, None)
        return future

    def _cancel_future(
        self,
        future: FutureLike,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        target_loop = loop
        if target_loop is None and isinstance(future, asyncio.Future):
            try:
                target_loop = future.get_loop()
            except Exception:
                target_loop = None
        if target_loop is not None:
            try:
                target_loop.call_soon_threadsafe(future.cancel)  # type: ignore[arg-type]
                return
            except Exception:
                pass
        try:
            future.cancel()
        except Exception:
            pass

    def _queue_on_loop(self, attr: str) -> asyncio.Queue[Any]:
        self.start()
        loop = self._loop
        subscriber_loop: asyncio.AbstractEventLoop | None = None
        try:
            subscriber_loop = asyncio.get_running_loop()
        except RuntimeError:
            subscriber_loop = None
        if self._owns_loop:
            assert loop is not None
            if subscriber_loop is None:
                if threading.current_thread() is self._loop_thread:
                    q: asyncio.Queue[Any] = asyncio.Queue()
                    self._register_queue(attr, q, loop)
                    return q

                async def _create() -> asyncio.Queue[Any]:
                    q_inner: asyncio.Queue[Any] = asyncio.Queue()
                    self._register_queue(attr, q_inner, loop)
                    return q_inner

                future = asyncio.run_coroutine_threadsafe(_create(), loop)
                return future.result()
            q = asyncio.Queue()
            self._register_queue(attr, q, subscriber_loop)
            return q

        target_loop = subscriber_loop or loop
        if target_loop is None:
            target_loop = asyncio.get_event_loop()
        q = asyncio.Queue()
        self._register_queue(attr, q, target_loop)
        return q

    def _register_queue(
        self,
        attr: str,
        queue: asyncio.Queue[Any],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        getattr(self, attr).add((queue, loop))

    def subscribe(self) -> asyncio.Queue[str]:
        """Return a queue that receives capability tier updates."""

        return self._queue_on_loop("_subscribers")

    def subscribe_usage(self) -> asyncio.Queue[Dict[str, float]]:
        """Return a queue that receives periodic resource usage samples."""

        return self._queue_on_loop("_usage_subscribers")

    async def _notify(self, tier: str) -> None:
        for q, loop in list(self._subscribers):
            try:
                loop.call_soon_threadsafe(q.put_nowait, tier)
            except Exception:
                self._subscribers.discard((q, loop))

    async def _notify_usage(self, usage: Dict[str, float]) -> None:
        for q, loop in list(self._usage_subscribers):
            try:
                loop.call_soon_threadsafe(q.put_nowait, usage)
            except Exception:
                self._usage_subscribers.discard((q, loop))

    def stop(self) -> None:
        cancelled: list[tuple[FutureLike, Optional[asyncio.AbstractEventLoop]]] = []
        for attr in ("_task", "_watch_task"):
            task = getattr(self, attr, None)
            if task is None:
                continue
            self._cancel_future(task, self._loop)
            cancelled.append((task, self._loop))
            setattr(self, attr, None)

        with self._tasks_lock:
            tracked = list(self._tasks.items())
            self._tasks.clear()
        for future, loop in tracked:
            self._cancel_future(future, loop)
        cancelled.extend(tracked)

        self._wait_for_shutdown(cancelled)

        def _drain(queue: asyncio.Queue[Any]) -> None:
            try:
                while True:
                    queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

        for attr_name in ("_subscribers", "_usage_subscribers"):
            queues = getattr(self, attr_name)
            for queue, owner_loop in list(queues):
                try:
                    owner_loop.call_soon_threadsafe(_drain, queue)
                except Exception:
                    try:
                        _drain(queue)
                    except Exception:
                        pass
            queues.clear()

        if self._owns_loop and self._loop is not None:
            self._drain_loop_tasks(self._loop)
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass
            if self._loop_thread and self._loop_thread.is_alive():
                self._loop_thread.join()
            try:
                self._loop.close()
            except Exception:
                pass
        self._loop = None
        self._loop_thread = None
        self._owns_loop = False

    def _wait_for_shutdown(
        self,
        cancelled: list[tuple[FutureLike, Optional[asyncio.AbstractEventLoop]]],
    ) -> None:
        if not cancelled:
            return

        loop_groups: Dict[asyncio.AbstractEventLoop, list[asyncio.Future[Any]]] = {}
        thread_futures: list[ThreadFuture] = []
        other_futures: list[FutureLike] = []

        for future, loop in cancelled:
            if isinstance(future, ThreadFuture):
                thread_futures.append(future)
                continue
            if isinstance(future, asyncio.Future):
                target_loop = loop
                if target_loop is None:
                    try:
                        target_loop = future.get_loop()
                    except Exception:
                        target_loop = None
                if target_loop is not None:
                    loop_groups.setdefault(target_loop, []).append(future)
                    continue
            other_futures.append(future)

        for loop, futures in loop_groups.items():
            if not futures:
                continue
            if loop.is_closed():
                continue
            gather_coro = asyncio.gather(*futures, return_exceptions=True)
            try:
                if loop.is_running():
                    try:
                        running_loop = asyncio.get_running_loop()
                    except RuntimeError:
                        running_loop = None
                    if running_loop is loop:
                        loop.create_task(gather_coro)
                    else:
                        future = asyncio.run_coroutine_threadsafe(gather_coro, loop)
                        future.result()
                else:
                    loop.run_until_complete(gather_coro)
            except Exception:
                pass

        for future in thread_futures:
            try:
                future.result()
            except Exception:
                pass

        for future in other_futures:
            try:
                if hasattr(future, "result"):
                    future.result()  # type: ignore[misc]
            except Exception:
                pass

    def _drain_loop_tasks(self, loop: asyncio.AbstractEventLoop) -> None:
        if loop.is_closed():
            return
        try:
            pending = asyncio.run_coroutine_threadsafe(
                self._gather_pending(loop), loop
            )
            pending.result()
        except Exception:
            pass

    async def _gather_pending(
        self, loop: asyncio.AbstractEventLoop
    ) -> None:  # pragma: no cover - shutdown helper
        current = asyncio.current_task()
        tasks = [t for t in asyncio.all_tasks() if t is not current]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

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
