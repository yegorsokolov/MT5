import pandas as pd
import pandas as pd
import pandas as pd
import pytest
import sys
from pathlib import Path
import asyncio

sys.path.append(str(Path(__file__).resolve().parents[1]))

import types


class _DummySpan:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class _DummyTracer:
    def start_as_current_span(self, *a, **k):
        return _DummySpan()


class _DummyMeter:
    def create_counter(self, *a, **k):
        return types.SimpleNamespace(add=lambda *a, **k: None)


sys.modules.setdefault(
    "telemetry",
    types.SimpleNamespace(get_tracer=lambda *a, **k: _DummyTracer(), get_meter=lambda *a, **k: _DummyMeter()),
)

from mt5 import state_manager
from analysis import replay
from mt5 import model_registry
import importlib.util

rm_spec = importlib.util.spec_from_file_location(
    "resource_monitor", Path(__file__).resolve().parents[1] / "utils" / "resource_monitor.py"
)
rm_module = importlib.util.module_from_spec(rm_spec)
assert rm_spec and rm_spec.loader
rm_spec.loader.exec_module(rm_module)  # type: ignore
ResourceMonitor = rm_module.ResourceMonitor
ResourceCapabilities = rm_module.ResourceCapabilities


def _patch_state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(state_manager, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(state_manager, "_REPLAY_TS_FILE", tmp_path / "replay_ts.txt")
    monkeypatch.setattr(replay, "REPLAY_DIR", tmp_path)


def test_reprocess_trades_persists_timestamp(monkeypatch, tmp_path):
    _patch_state_dir(tmp_path, monkeypatch)
    decisions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"]),
            "event": ["prediction"],
            "Symbol": ["XYZ"],
            "prob": [0.5],
        }
    )
    dummy_log = types.SimpleNamespace(read_decisions=lambda: decisions)
    monkeypatch.setitem(sys.modules, "log_utils", dummy_log)
    calls = []
    dummy_replay = types.SimpleNamespace(replay_trades=lambda v: calls.append(v))
    monkeypatch.setitem(sys.modules, "analysis.replay_trades", dummy_replay)
    monkeypatch.setattr(model_registry, "select_models", lambda: ["m1"])

    replay.reprocess_trades()
    assert calls == [["m1"]]
    assert state_manager.load_replay_timestamp() == "2024-01-01T00:00:00"

    replay.reprocess_trades()
    assert calls == [["m1"]]

def test_probe_triggers_replay_on_upgrade(monkeypatch, tmp_path):
    _patch_state_dir(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(model_registry, "select_models", lambda: calls.append("select"))
    monkeypatch.setattr(replay, "reprocess_trades", lambda: calls.append("replay"))

    monitor = ResourceMonitor()
    monitor.capability_tier = "lite"
    monkeypatch.setattr(monitor, "_probe", lambda: ResourceCapabilities(8, 32, True, 1))

    asyncio.run(monitor.probe())
    assert calls == ["select", "replay"]

    asyncio.run(monitor.probe())
    assert calls == ["select", "replay"]
