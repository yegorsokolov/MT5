import time
from pathlib import Path
import importlib.util
import pandas as pd
import pytest
import types
import sys

stub_utils = types.ModuleType("utils")
stub_utils.resource_monitor = types.SimpleNamespace(monitor=types.SimpleNamespace(latest_usage={}))
sys.modules["utils"] = stub_utils
sys.modules["utils.resource_monitor"] = stub_utils.resource_monitor
stub_metrics = types.ModuleType("analytics.metrics_store")
stub_metrics.record_metric = lambda *a, **k: None
sys.modules.setdefault("analytics", types.ModuleType("analytics"))
sys.modules["analytics.metrics_store"] = stub_metrics

root = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(root))
spec = importlib.util.spec_from_file_location(
    "feature_store", root / "data" / "feature_store.py"
)
fs_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs_mod)  # type: ignore[attr-defined]
FeatureStore = fs_mod.FeatureStore


def test_memory_tier_hit_ratio(monkeypatch, tmp_path):
    metrics = []
    monkeypatch.setattr(fs_mod, "record_metric", lambda name, value, tags=None: metrics.append((name, value, tags)))
    fs = FeatureStore(path=tmp_path / "store.duckdb", memory_size=2)

    df = pd.DataFrame({"a": [1, 2, 3]})

    def compute():
        return df

    res1 = fs.get_features("AAA", "2020-01-01", "2020-01-02", compute)
    time.sleep(0.1)
    metrics.clear()
    res2 = fs.get_features("AAA", "2020-01-01", "2020-01-02", compute)
    time.sleep(0.1)

    mem_metrics = [m for m in metrics if m[2].get("tier") == "memory"]
    assert mem_metrics[-1][1] == pytest.approx(0.5)
    pd.testing.assert_frame_equal(res1, res2)
