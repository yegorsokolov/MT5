import asyncio
import importlib
import sys
import types
from dataclasses import dataclass


def test_import_side_effects_and_lazy_capability_watch(monkeypatch):
    """Importing :mod:`features` should not touch the monitor."""

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

        def subscribe(self):
            calls["subscribe"] += 1
            return asyncio.Queue()

        def create_task(self, coro):
            calls["create_task"] += 1
            coro.close()
            return types.SimpleNamespace(task=coro)

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

    cross_spectral = types.ModuleType("analysis.cross_spectral")
    cross_spectral.compute = lambda df, *_args, **_kwargs: df
    analysis_pkg = types.ModuleType("analysis")
    analysis_pkg.__path__ = []
    analysis_pkg.cross_spectral = cross_spectral
    monkeypatch.setitem(sys.modules, "analysis", analysis_pkg)
    monkeypatch.setitem(sys.modules, "analysis.cross_spectral", cross_spectral)

    features = importlib.import_module("features")

    assert calls == {"subscribe": 0, "create_task": 0}

    features.start_capability_watch()
    features.start_capability_watch()

    assert calls == {"subscribe": 1, "create_task": 1}
