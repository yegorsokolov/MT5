import importlib.util
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Import plugin_runner directly to avoid executing utils.__init__
spec = importlib.util.spec_from_file_location(
    "plugin_runner", REPO_ROOT / "utils" / "plugin_runner.py"
)
plugin_runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(plugin_runner)
run_plugin = plugin_runner.run_plugin
PluginTimeoutError = plugin_runner.PluginTimeoutError


def _hung_plugin():
    while True:
        time.sleep(0.1)


def _safe_plugin():
    return "ok"


def test_hung_plugin_does_not_block():
    with pytest.raises(PluginTimeoutError):
        run_plugin(_hung_plugin, timeout=0.5)
    assert run_plugin(_safe_plugin, timeout=1) == "ok"
