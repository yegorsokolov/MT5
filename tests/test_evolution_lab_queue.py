import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

# Stub services.message_bus to satisfy ShadowRunner import
bus_mod = types.ModuleType("services.message_bus")

class _DummyBus:
    async def subscribe(self, *_args, **_kwargs):  # pragma: no cover - not used
        if False:
            yield {}

def _get_bus():  # pragma: no cover - not used
    return _DummyBus()

class _Topics:
    SIGNALS = "signals"

bus_mod.get_message_bus = _get_bus
bus_mod.MessageBus = _DummyBus
bus_mod.Topics = _Topics
services_pkg = types.ModuleType("services")
services_pkg.message_bus = bus_mod
sys.modules.setdefault("services", services_pkg)
sys.modules.setdefault("services.message_bus", bus_mod)

# Provide a minimal "strategy" package so EvolutionLab's relative imports resolve
pkg = types.ModuleType("strategy")
pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "strategy")]
sys.modules.setdefault("strategy", pkg)

# Load EvolutionLab dynamically to avoid package side effects
spec = importlib.util.spec_from_file_location(
    "strategy.evolution_lab",
    Path(__file__).resolve().parents[1] / "strategy" / "evolution_lab.py",
)
evo_mod = importlib.util.module_from_spec(spec)
sys.modules["strategy.evolution_lab"] = evo_mod
spec.loader.exec_module(evo_mod)  # type: ignore
EvolutionLab = evo_mod.EvolutionLab


def _messages():
    return [
        {"Timestamp": 1, "x": 0.1},
        {"Timestamp": 2, "x": -0.2},
        {"Timestamp": 3, "x": 0.3},
    ]


def _base(msg):
    return float(msg["x"])


def test_evolve_queue_runs_multiple_jobs():
    async def _run():
        lab = EvolutionLab(_base)
        results = await lab.evolve_queue([0, 1], _messages())
        assert len(results) == 2
        names = {r["name"] for r in results}
        assert names == {"_base_0", "_base_1"}
        for res in results:
            assert "sharpe" in res

    asyncio.run(_run())
