import sys

import pandas as pd
import pytest

from plugins import PLUGIN_SPECS, FEATURE_PLUGINS, monitor, _rm

ResourceCapabilities = _rm.ResourceCapabilities


def _gpu_spec():
    for spec in PLUGIN_SPECS:
        if spec.name == "gpu_feature":
            return spec
    raise AssertionError("gpu_feature spec not found")


def test_gpu_plugin_deferred_on_lite(monkeypatch, caplog):
    spec = _gpu_spec()
    FEATURE_PLUGINS.clear()
    sys.modules.pop("plugins.gpu_feature", None)
    monkeypatch.setattr(
        monitor,
        "capabilities",
        ResourceCapabilities(cpus=1, memory_gb=1, has_gpu=False, gpu_count=0),
    )
    with caplog.at_level("INFO"):
        result = spec.load()
    assert result is None
    assert not FEATURE_PLUGINS
    assert "Skipping plugin gpu_feature" in "\n".join(caplog.messages)


def test_gpu_plugin_loaded_on_gpu(monkeypatch):
    spec = _gpu_spec()
    FEATURE_PLUGINS.clear()
    sys.modules.pop("plugins.gpu_feature", None)
    monkeypatch.setattr(
        monitor,
        "capabilities",
        ResourceCapabilities(cpus=4, memory_gb=8, has_gpu=True, gpu_count=1),
    )
    mod = spec.load()
    assert mod is not None
    assert any(f.__name__ == "gpu_feature" for f in FEATURE_PLUGINS)
