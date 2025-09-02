import json
from pathlib import Path

from utils.resource_monitor import ResourceCapabilities


def test_report_status(monkeypatch, tmp_path):
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
    assert report["suggestion"]

    saved = json.loads((Path(tmp_path) / "latest.json").read_text())
    assert saved == report

