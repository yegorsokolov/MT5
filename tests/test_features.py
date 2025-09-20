import asyncio
import importlib
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


class _FakeTask:
    def __init__(self, coro) -> None:
        self._coro = coro
        self._callbacks = []
        self._done = False
        self._cancelled = False

    def add_done_callback(self, callback):
        self._callbacks.append(callback)
        if self._done:
            callback(self)

    def cancel(self) -> bool:
        if self._done:
            return False
        self._cancelled = True
        self._finish()
        return True

    def finish(self) -> None:
        if self._done:
            return
        self._finish()

    def done(self) -> bool:
        return self._done

    def _finish(self) -> None:
        self._done = True
        try:
            self._coro.close()
        except RuntimeError:
            pass
        for callback in list(self._callbacks):
            callback(self)


def _import_features(monkeypatch) -> Tuple[types.ModuleType, Dict[str, int]]:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    for name in list(sys.modules):
        if name == "features" or name.startswith("features."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules, "analysis", raising=False)
    monkeypatch.delitem(sys.modules, "analysis.cross_spectral", raising=False)
    monkeypatch.delitem(sys.modules, "data", raising=False)
    monkeypatch.delitem(sys.modules, "data.features", raising=False)

    calls = {"subscribe": 0, "create_task": 0}

    @dataclass
    class _Caps:
        cpus: int
        memory_gb: float
        has_gpu: bool
        gpu_count: int

    class _Monitor:
        def __init__(self) -> None:
            self.capabilities = _Caps(cpus=2, memory_gb=4.0, has_gpu=False, gpu_count=0)
            self.tasks = []

        def subscribe(self):
            calls["subscribe"] += 1
            return asyncio.Queue()

        def create_task(self, coro):
            calls["create_task"] += 1
            task = _FakeTask(coro)
            self.tasks.append(task)
            return task

    resource_monitor = types.ModuleType("utils.resource_monitor")
    resource_monitor.ResourceCapabilities = _Caps
    resource_monitor.monitor = _Monitor()

    utils_mod = types.ModuleType("utils")
    utils_mod.load_config = lambda: {}
    utils_mod.resource_monitor = resource_monitor

    monkeypatch.setitem(sys.modules, "utils", utils_mod)
    monkeypatch.setitem(sys.modules, "utils.resource_monitor", resource_monitor)

    data_pkg = types.ModuleType("data")
    data_pkg.__path__ = []
    data_features = types.ModuleType("data.features")
    data_features.make_features = lambda *a, **k: None
    data_pkg.features = data_features
    monkeypatch.setitem(sys.modules, "data", data_pkg)
    monkeypatch.setitem(sys.modules, "data.features", data_features)

    stub_names = [
        "price",
        "news",
        "cross_asset",
        "orderbook",
        "order_flow",
        "microprice",
        "liquidity_exhaustion",
        "auto_indicator",
        "volume",
        "multi_timeframe",
        "supertrend",
        "keltner_squeeze",
        "adaptive_ma",
        "kalman_ma",
        "regime",
        "macd",
        "ram",
        "cointegration",
        "vwap",
        "baseline_signal",
        "divergence",
        "evolved_indicators",
        "evolved_symbols",
        "oracle_intelligence",
    ]
    for name in stub_names:
        mod = types.ModuleType(f"features.{name}")
        mod.compute = lambda df, *_args, **_kwargs: df
        monkeypatch.setitem(sys.modules, f"features.{name}", mod)

    validators_mod = types.ModuleType("features.validators")
    validators_mod.validate_ge = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "features.validators", validators_mod)

    cross_spectral = types.ModuleType("analysis.cross_spectral")
    cross_spectral.compute = lambda df, *_args, **_kwargs: df
    analysis_pkg = types.ModuleType("analysis")
    analysis_pkg.__path__ = []
    analysis_pkg.cross_spectral = cross_spectral
    monkeypatch.setitem(sys.modules, "analysis", analysis_pkg)
    monkeypatch.setitem(sys.modules, "analysis.cross_spectral", cross_spectral)

    monkeypatch.setenv("MT5_DOCS_BUILD", "1")
    features = importlib.import_module("features")
    monkeypatch.delenv("MT5_DOCS_BUILD", raising=False)

    return features, calls


def test_import_side_effects_and_lazy_capability_watch(monkeypatch):
    """Importing :mod:`features` should not touch the monitor."""

    features, calls = _import_features(monkeypatch)

    assert calls == {"subscribe": 0, "create_task": 0}

    features.start_capability_watch()
    features.start_capability_watch()

    assert calls == {"subscribe": 1, "create_task": 1}

    # Ensure the coroutine is cleaned up for the next test run.
    task = features.monitor.tasks[0]
    assert features._capability_watch_future is task
    task.cancel()
    shutil.rmtree("reports/feature_status", ignore_errors=True)


def test_capability_watch_restarts_after_task_cancellation(monkeypatch):
    features, calls = _import_features(monkeypatch)

    features.start_capability_watch()
    first_task = features._capability_watch_future
    assert first_task is not None
    assert first_task is features.monitor.tasks[0]

    # Simulate monitor.stop() cancelling the task.
    assert first_task.cancel()

    features.start_capability_watch()
    assert calls == {"subscribe": 2, "create_task": 2}
    second_task = features._capability_watch_future
    assert second_task is not None
    assert second_task is features.monitor.tasks[-1]
    assert second_task is not first_task

    # Repeated start/stop cycles should also succeed.
    assert second_task.cancel()
    features.start_capability_watch()
    assert calls == {"subscribe": 3, "create_task": 3}
    third_task = features._capability_watch_future
    assert third_task is not None
    assert third_task is features.monitor.tasks[-1]
    assert third_task is not second_task

    # Cleanup for the last created task.
    third_task.cancel()
    shutil.rmtree("reports/feature_status", ignore_errors=True)
