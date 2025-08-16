import asyncio
import importlib.util
from pathlib import Path

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import psutil
import pytest
import analytics.metrics_store as ms

spec = importlib.util.spec_from_file_location(
    "resource_monitor", Path(__file__).resolve().parents[1] / "utils" / "resource_monitor.py"
)
rm_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm_mod)
ResourceMonitor = rm_mod.ResourceMonitor


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

    calls = []
    def fake_record(name, value, tags=None):
        calls.append((name, value))
    monkeypatch.setattr(ms, "record_metric", fake_record)
    monkeypatch.setattr(rm_mod, "record_metric", fake_record)

    triggered = asyncio.Event()

    async def on_breach(reason: str):
        fake_record("resource_restarts", 1)
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
    assert any(c[0] == "resource_restarts" for c in calls)

