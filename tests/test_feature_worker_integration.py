import importlib.util
from pathlib import Path
import types
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from starlette.responses import Response

# Stub out optional modules used by FeatureStore
stub_utils = types.ModuleType("utils")
stub_utils.resource_monitor = types.SimpleNamespace(
    monitor=types.SimpleNamespace(latest_usage={}, capability_tier="lite")
)
sys.modules["utils"] = stub_utils
sys.modules["utils.resource_monitor"] = stub_utils.resource_monitor

stub_metrics = types.ModuleType("analytics.metrics_store")
stub_metrics.record_metric = lambda *a, **k: None
sys.modules.setdefault("analytics", types.ModuleType("analytics"))
sys.modules["analytics.metrics_store"] = stub_metrics

root = Path(__file__).resolve().parents[1]

# Load FeatureStore module dynamically to use stubs above
spec = importlib.util.spec_from_file_location(
    "feature_store", root / "data" / "feature_store.py"
)
fs_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs_mod)  # type: ignore[attr-defined]
FeatureStore = fs_mod.FeatureStore

# Import worker service
spec_worker = importlib.util.spec_from_file_location(
    "feature_worker", root / "services" / "feature_worker.py"
)
worker_mod = importlib.util.module_from_spec(spec_worker)
spec_worker.loader.exec_module(worker_mod)  # type: ignore[attr-defined]
app = worker_mod.app


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_feature_store_uses_remote_worker_on_lite_host(monkeypatch, tmp_path):
    """Ensure remote worker is invoked on lite hosts and results are cached."""

    client = TestClient(app)
    calls = {"count": 0}

    def fake_post(url, json=None, timeout=30):
        calls["count"] += 1
        if calls["count"] == 1:
            # First call fails to trigger retry logic
            return Response(status_code=500)
        return client.post("/compute", json=json)

    monkeypatch.setattr(
        fs_mod,
        "requests",
        types.SimpleNamespace(post=fake_post),
    )

    fs = FeatureStore(path=tmp_path / "store.duckdb", worker_url="http://testserver")
    results = {"local": 0}

    def compute_local():
        results["local"] += 1
        s, e = 0, 3
        return pd.DataFrame({
            "symbol": ["AAA"] * (e - s),
            "value": [i * i for i in range(s, e)],
        })

    df1 = fs.get_features("AAA", "0", "3", compute_local)
    df2 = fs.get_features("AAA", "0", "3", compute_local)

    assert results["local"] == 0  # remote worker used
    assert calls["count"] == 2  # one failure + one success, cached afterwards
    expected = pd.DataFrame({"symbol": ["AAA", "AAA", "AAA"], "value": [0, 1, 4]})
    pd.testing.assert_frame_equal(df1, expected)
    pd.testing.assert_frame_equal(df2, expected)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_feature_store_local_compute_on_standard_host(monkeypatch, tmp_path):
    """Ensure local compute is used when resources are sufficient."""

    stub_utils.resource_monitor.monitor.capability_tier = "standard"
    client = TestClient(app)
    calls = {"count": 0}

    def fake_post(url, json=None, timeout=30):
        calls["count"] += 1
        return client.post("/compute", json=json)

    monkeypatch.setattr(
        fs_mod,
        "requests",
        types.SimpleNamespace(post=fake_post),
    )

    fs = FeatureStore(path=tmp_path / "store.duckdb", worker_url="http://testserver")
    results = {"local": 0}

    def compute_local():
        results["local"] += 1
        s, e = 0, 3
        return pd.DataFrame({
            "symbol": ["AAA"] * (e - s),
            "value": [i * i for i in range(s, e)],
        })

    df = fs.get_features("AAA", "0", "3", compute_local)
    assert results["local"] == 1
    assert calls["count"] == 0  # worker not used
    expected = pd.DataFrame({"symbol": ["AAA", "AAA", "AAA"], "value": [0, 1, 4]})
    pd.testing.assert_frame_equal(df, expected)
