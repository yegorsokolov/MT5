import importlib
import sys
import time
from pathlib import Path
import types

# Stub external dependencies before importing metrics/plugins
class _DummyMetric:
    def __init__(self, *a, **k):
        self.count = 0

    def inc(self, amount: int = 1):  # pragma: no cover - trivial
        self.count += amount

    def set(self, *a, **k):  # pragma: no cover - unused
        pass

prom_stub = types.SimpleNamespace(Counter=_DummyMetric, Gauge=_DummyMetric)
sys.modules.setdefault("prometheus_client", prom_stub)

class _DummyRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - trivial
        return False

mlflow_stub = types.SimpleNamespace(
    set_tracking_uri=lambda *a, **k: None,
    set_experiment=lambda *a, **k: None,
    start_run=lambda *a, **k: _DummyRun(),
    log_dict=lambda *a, **k: None,
)
sys.modules.setdefault("mlflow", mlflow_stub)

# Minimal stubs for optional dependencies

from mt5 import metrics
metrics.PLUGIN_RELOADS = _DummyMetric()
REPO_ROOT = Path(__file__).resolve().parents[1]

# Create a minimal plugins package stub used by hot reloading
plugins_stub = types.ModuleType("plugins")
plugins_stub.__path__ = [str(REPO_ROOT / "plugins")]  # type: ignore[attr-defined]
plugins_stub.FEATURE_PLUGINS = []

def _reg(func=None, *, name=None, tier="lite"):
    def decorator(f):
        plugins_stub.FEATURE_PLUGINS.append(
            types.SimpleNamespace(name=name or f.__name__, plugin=f)
        )
        return f
    if func is None:
        return decorator
    return decorator(func)

plugins_stub.register_feature = _reg  # type: ignore[attr-defined]
sys.modules["plugins"] = plugins_stub

# Import plugin_runner directly to avoid utils.__init__ side effects
spec = importlib.util.spec_from_file_location(
    "plugin_runner", REPO_ROOT / "utils" / "plugin_runner.py"
)
plugin_runner = importlib.util.module_from_spec(spec)  # noqa: F841
assert spec.loader is not None
spec.loader.exec_module(plugin_runner)

plugins = plugins_stub

# Allow watcher thread to start
time.sleep(0.2)
PLUGIN_FILE = REPO_ROOT / "plugins" / "_temp_live.py"
PLUGIN_FILE.unlink(missing_ok=True)

def _write_plugin(version: int, broken: bool = False) -> None:
    if broken:
        PLUGIN_FILE.write_text("def broken(:\n")
        return
    PLUGIN_FILE.write_text(
        """
from plugins import register_feature

VERSION = {version}

@register_feature(name="temp_live")
def temp_live():
    return VERSION
""".format(version=version)
    )


def _get_spec():
    for spec in plugins.FEATURE_PLUGINS:
        if spec.name == "temp_live":
            return spec.plugin
    raise AssertionError("plugin not registered")


def _wait(cond, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.1)
    return False


def test_plugin_hot_reload():
    _write_plugin(1)
    assert _wait(lambda: metrics.PLUGIN_RELOADS.count >= 1)
    assert _wait(lambda: _get_spec()() == 1)
    first = metrics.PLUGIN_RELOADS.count

    time.sleep(1.1)
    _write_plugin(2)
    assert _wait(lambda: _get_spec()() == 2)
    assert metrics.PLUGIN_RELOADS.count > first

    prev = metrics.PLUGIN_RELOADS.count
    time.sleep(1.1)
    _write_plugin(3, broken=True)
    time.sleep(0.5)
    assert _get_spec()() == 2
    assert metrics.PLUGIN_RELOADS.count == prev

    PLUGIN_FILE.unlink(missing_ok=True)
