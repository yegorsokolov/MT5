import importlib.util
import sys
import threading
import types
from pathlib import Path

import pytest


proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

# Provide a tiny numpy stub to satisfy ShadowRunner imports
if "numpy" not in sys.modules:
    numpy_stub = types.ModuleType("numpy")

    def _identity_array(data, dtype=None):  # pragma: no cover - simple stub
        return data

    def _mean(arr):  # pragma: no cover - not used in test
        return sum(arr) / len(arr) if arr else 0.0

    def _std(arr, ddof=0):  # pragma: no cover - not used in test
        return 0.0

    numpy_stub.array = _identity_array
    numpy_stub.mean = _mean
    numpy_stub.std = _std
    sys.modules["numpy"] = numpy_stub

# Provide a minimal ``strategy`` package namespace so relative imports succeed
pkg = types.ModuleType("strategy")
pkg.__path__ = [str(proj_root / "strategy")]
sys.modules.setdefault("strategy", pkg)

spec = importlib.util.spec_from_file_location(
    "strategy.evolution_lab",
    proj_root / "strategy" / "evolution_lab.py",
)
evo_mod = importlib.util.module_from_spec(spec)
sys.modules["strategy.evolution_lab"] = evo_mod
assert spec and spec.loader
spec.loader.exec_module(evo_mod)  # type: ignore[attr-defined]
EvolutionLab = evo_mod.EvolutionLab


@pytest.fixture(autouse=True)
def cleanup_background_loop():
    evo_mod._shutdown_background_loop()
    yield
    evo_mod._shutdown_background_loop()


def _base_strategy(features):
    return float(features["value"])


def test_spawn_without_orchestrator_runs_shadow_runner(monkeypatch):
    class RecordingShadowRunner:
        instances = []
        next_message = {"Timestamp": 1, "value": 0.0}

        def __init__(self, name, handler, **_kwargs):
            self.name = name
            self.handler = handler
            self.processed = []
            self.done = threading.Event()
            RecordingShadowRunner.instances.append(self)

        async def run(self):
            msg = dict(RecordingShadowRunner.next_message)
            result = float(self.handler(msg))
            self.processed.append(result)
            self.done.set()

    RecordingShadowRunner.instances.clear()
    monkeypatch.setattr(evo_mod, "ShadowRunner", RecordingShadowRunner)

    lab = EvolutionLab(_base_strategy)

    def _mutate_stub(self, _scale: float = 0.1, *, with_info: bool = False):
        if with_info:
            return self.base, {"type": "noop"}
        return self.base

    monkeypatch.setattr(
        lab,
        "_mutate",
        types.MethodType(_mutate_stub, lab),
        raising=False,
    )

    RecordingShadowRunner.next_message = {"Timestamp": 1, "value": 2.5}
    lab.spawn(1)
    assert RecordingShadowRunner.instances, "runner should be spawned"
    runner1 = RecordingShadowRunner.instances[-1]
    assert runner1.done.wait(timeout=1.0)
    assert runner1.processed == [2.5]
    thread = evo_mod._BACKGROUND_THREAD
    assert thread is not None and thread.is_alive()

    RecordingShadowRunner.next_message = {"Timestamp": 2, "value": -1.5}
    lab.spawn(1)
    assert len(RecordingShadowRunner.instances) == 2
    runner2 = RecordingShadowRunner.instances[-1]
    assert runner2 is not runner1
    assert runner2.done.wait(timeout=1.0)
    assert runner2.processed == [-1.5]
    assert evo_mod._BACKGROUND_THREAD is thread
    assert thread.is_alive()
