import asyncio
import importlib.util
from pathlib import Path

import psutil
from prometheus_client import Counter, CollectorRegistry
import pytest


spec = importlib.util.spec_from_file_location(
    "resource_monitor", Path(__file__).resolve().parents[1] / "utils" / "resource_monitor.py"
)
rm_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm_mod)
ResourceMonitor = rm_mod.ResourceMonitor

spec = importlib.util.spec_from_file_location(
    "metrics", Path(__file__).resolve().parents[1] / "metrics.py"
)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


@pytest.mark.asyncio
async def test_watchdog_triggers(monkeypatch):
    class FakeProc:
        def memory_info(self):
            class M:
                rss = 100 * 1024 * 1024

            return M()

        def cpu_percent(self):
            return 100.0

    monkeypatch.setattr(psutil, "Process", lambda: FakeProc())

    registry = CollectorRegistry()
    rr = Counter("resource_restarts", "x", registry=registry)
    monkeypatch.setattr(metrics, "RESOURCE_RESTARTS", rr, raising=False)

    triggered = asyncio.Event()

    async def on_breach(reason: str):
        metrics.RESOURCE_RESTARTS.inc()
        triggered.set()

    monitor = ResourceMonitor(
        max_rss_mb=1,
        max_cpu_pct=1,
        sample_interval=0.05,
        breach_duration=0.1,
        alert_callback=on_breach,
    )
    monitor.start()
    await asyncio.wait_for(triggered.wait(), timeout=1)
    monitor.stop()
    assert rr._value.get() == 1

