import sys
import types
from pathlib import Path

# ensure repo root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# stub analytics metrics store to capture metric names
_metric_calls = []
metrics_stub = types.SimpleNamespace(record_metric=lambda name, value, tags=None: _metric_calls.append(name))
sys.modules["analytics.metrics_store"] = metrics_stub

# provide minimal utils stub for plugin imports
sys.modules.setdefault("utils", types.SimpleNamespace(load_config=lambda: {}))

import plugins


def test_plugin_eviction_and_reload(monkeypatch):
    _metric_calls.clear()
    spec = next(p for p in plugins.PLUGIN_SPECS if p.name == "atr")
    t0 = 1000.0
    monkeypatch.setattr(plugins.time, "time", lambda: t0)
    mod1 = spec.load()
    assert spec._loaded and mod1 is not None
    monkeypatch.setattr(plugins.time, "time", lambda: t0 + 2)
    plugins.purge_unused_plugins(ttl=1)
    assert spec._loaded is False
    assert "plugins.atr" not in sys.modules
    assert "plugin_unloads" in _metric_calls
    monkeypatch.setattr(plugins.time, "time", lambda: t0 + 3)
    mod2 = spec.load()
    assert spec._loaded and mod2 is not None
    assert mod1 is not mod2
