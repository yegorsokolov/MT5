import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.performance_monitor import PerformanceMonitor
from scheduler import process_retrain_events
import importlib.util

spec = importlib.util.spec_from_file_location(
    "analytics.metrics_store",
    Path(__file__).resolve().parents[1] / "analytics" / "metrics_store.py",
)
metrics_store = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics_store)  # type: ignore
import types, sys
pkg = types.ModuleType("analytics")
pkg.metrics_store = metrics_store
sys.modules["analytics"] = pkg
sys.modules["analytics.metrics_store"] = metrics_store


def test_retrain_scheduled_on_drift(tmp_path, monkeypatch):
    class DummyStore:
        def __init__(self) -> None:
            self.events = []

        def record(self, event_type: str, payload: dict) -> None:
            self.events.append({"timestamp": str(len(self.events)), "type": event_type, "payload": payload})

        def iter_events(self, event_type: str):
            return [e for e in self.events if e["type"] == event_type]

    store = DummyStore()
    monitor = PerformanceMonitor(
        pnl_threshold=-1.0,
        drift_threshold=0.5,
        model="nn",
        store=store,
    )

    monitor.record(0.0, 0.0)
    monitor.record(0.0, 1.0)

    assert list(store.iter_events("retrain"))

    calls = []

    def fake_run(cmd, check, env=None):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    logged: list[tuple[str, str]] = []
    monkeypatch.setattr(metrics_store, "log_retrain_outcome", lambda m, s: logged.append((m, s)))

    process_retrain_events(store)

    assert calls and any("train_cli.py" in part for part in calls[0])
    assert logged and logged[0] == ("nn", "success")
