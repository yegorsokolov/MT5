import json
from pathlib import Path
import types
import sys


def test_report_status(monkeypatch, tmp_path):
    rm_stub = types.ModuleType("utils.resource_monitor")

    class ResourceCapabilities:
        def __init__(self, cpus, memory_gb, has_gpu, gpu_count):
            self.cpus = cpus
            self.memory_gb = memory_gb
            self.has_gpu = has_gpu
            self.gpu_count = gpu_count

    rm_stub.ResourceCapabilities = ResourceCapabilities
    rm_stub.monitor = types.SimpleNamespace(
        capabilities=ResourceCapabilities(1, 2, False, 0),
        subscribe=lambda: types.SimpleNamespace(),
    )

    utils_stub = types.ModuleType("utils")
    utils_stub.load_config = lambda: {}
    utils_stub.resource_monitor = rm_stub
    sys.modules["utils"] = utils_stub
    sys.modules["utils.resource_monitor"] = rm_stub

    # Stub feature modules to avoid heavy dependencies
    for name in ["price", "news", "cross_asset", "orderbook", "auto_indicator"]:
        mod = types.ModuleType(f"features.{name}")
        mod.compute = lambda df, *_args, **_kwargs: df
        sys.modules[f"features.{name}"] = mod

    import features

    # Redirect report output to temporary directory
    monkeypatch.setattr(features, "_REPORT_DIR", Path(tmp_path))

    # Simulate limited resources
    caps = ResourceCapabilities(cpus=1, memory_gb=2, has_gpu=False, gpu_count=0)
    monkeypatch.setattr(features.monitor, "capabilities", caps)

    features._update_status()
    report = features.report_status()

    statuses = {f["name"]: f["status"] for f in report["features"]}
    assert statuses["price"] == "active"
    assert statuses["news"] == "skipped_insufficient_resources"
    assert statuses["cross_asset"] == "skipped_insufficient_resources"
    assert statuses["orderbook"] == "skipped_insufficient_resources"
    assert statuses["auto_indicator"] == "active"
    assert report["suggestion"]

    saved = json.loads((Path(tmp_path) / "latest.json").read_text())
    assert saved == report

