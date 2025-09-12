import importlib.util
import random
import sys
from pathlib import Path

import pytest
import types

# Create lightweight "services.message_bus" stub required by ShadowRunner
bus_mod = types.ModuleType("services.message_bus")


class _DummyBus:
    async def subscribe(self, *_, **__):  # pragma: no cover - not used in test
        if False:
            yield {}


def _get_bus() -> _DummyBus:  # pragma: no cover - not used in test
    return _DummyBus()


class _Topics:
    SIGNALS = "signals"


bus_mod.get_message_bus = _get_bus
bus_mod.Topics = _Topics
bus_mod.MessageBus = _DummyBus

services_pkg = types.ModuleType("services")
services_pkg.message_bus = bus_mod
sys.modules.setdefault("services", services_pkg)
sys.modules.setdefault("services.message_bus", bus_mod)

# Create lightweight "strategy" package to satisfy relative imports
pkg = types.ModuleType("strategy")
pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "strategy")]
sys.modules.setdefault("strategy", pkg)

spec = importlib.util.spec_from_file_location(
    "strategy.evolution_lab",
    Path(__file__).resolve().parents[1] / "strategy" / "evolution_lab.py",
)
evo_mod = importlib.util.module_from_spec(spec)
sys.modules["strategy.evolution_lab"] = evo_mod
assert spec and spec.loader
spec.loader.exec_module(evo_mod)  # type: ignore
EvolutionLab = evo_mod.EvolutionLab


def test_mutate_mix_creates_composite_strategy(monkeypatch):
    """EvolutionLab should blend strategies to form more complex variants."""

    # Base strategy simply returns the provided value
    def base(features):
        return float(features["x"])

    lab = EvolutionLab(base)

    # Ensure initial variant adds a constant offset for deterministic behaviour
    monkeypatch.setattr(random, "gauss", lambda *a, **k: 1.0)
    lab.spawn(1)

    calls = {"i": 0}

    def fake_choice(seq):
        # First call chooses the 'mix' operator, second selects existing variant
        if calls["i"] == 0:
            calls["i"] += 1
            return "mix"
        return seq[0]

    monkeypatch.setattr(random, "choice", fake_choice)
    monkeypatch.setattr(random, "random", lambda: 0.5)

    variant = lab._mutate()

    features = {"x": 2.0}
    base_val = base(features)
    other_val = list(lab.variants.values())[0](features)
    out = variant(features)

    assert out == pytest.approx(0.5 * base_val + 0.5 * other_val)
