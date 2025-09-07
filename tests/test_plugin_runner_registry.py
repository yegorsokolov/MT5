import sys
import time
from pathlib import Path

import pytest

# Ensure repository root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import plugins

PluginTimeoutError = plugins.PluginTimeoutError


def test_registry_hung_plugin():
    plugins.FEATURE_PLUGINS.clear()

    @plugins.register_feature
    def _hung(_: object = None):
        while True:
            time.sleep(0.1)

    wrapped = plugins.FEATURE_PLUGINS[0]
    with pytest.raises(PluginTimeoutError):
        wrapped(_timeout=0.5)

    plugins.FEATURE_PLUGINS.clear()


def test_registry_crashing_plugin():
    plugins.FEATURE_PLUGINS.clear()

    @plugins.register_feature
    def _crash(_: object = None):
        raise RuntimeError("boom")

    wrapped = plugins.FEATURE_PLUGINS[0]
    with pytest.raises(RuntimeError):
        wrapped()

    plugins.FEATURE_PLUGINS.clear()

    @plugins.register_feature
    def _safe(_: object = None):
        return "ok"

    assert plugins.FEATURE_PLUGINS[0]() == "ok"
