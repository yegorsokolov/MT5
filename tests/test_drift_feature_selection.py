import asyncio
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Load feature selector and drift monitor modules from repository root
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

fs_spec = importlib.util.spec_from_file_location(
    "feature_selector", root / "analysis" / "feature_selector.py"
)
feature_selector = importlib.util.module_from_spec(fs_spec)
fs_spec.loader.exec_module(feature_selector)

md_spec = importlib.util.spec_from_file_location(
    "monitor_drift", root / "monitor_drift.py"
)
monitor_drift = importlib.util.module_from_spec(md_spec)
sys.modules["monitor_drift"] = monitor_drift
md_spec.loader.exec_module(monitor_drift)


def test_feature_drift_updates_selected_features(tmp_path):
    # cancel global task from module import to avoid side effects
    if monitor_drift.monitor._task is not None:
        monitor_drift.monitor._task.cancel()
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0))

    rng = np.random.default_rng(0)
    baseline = pd.DataFrame({
        "f1": rng.normal(size=400),
        "f2": rng.normal(size=400),
    })
    baseline["target"] = (baseline["f1"] > 0).astype(int)
    baseline_path = tmp_path / "baseline.parquet"
    baseline.to_parquet(baseline_path, index=False)

    feature_file = tmp_path / "features.json"
    base_selected = feature_selector.select_features(
        baseline[["f1", "f2"]], baseline["target"]
    )
    base_version = feature_selector.save_feature_set(base_selected, feature_file)
    assert "f1" in base_selected and "f2" not in base_selected

    dm = monitor_drift.DriftMonitor(
        baseline_path=baseline_path,
        store_path=tmp_path / "curr.parquet",
        threshold=0.05,
        drift_threshold=0.1,
        feature_set_path=feature_file,
    )

    current = pd.DataFrame({
        "f1": rng.normal(loc=10.0, size=400),  # strong drift
        "f2": rng.normal(size=400),
    })
    current["target"] = (current["f2"] > 0).astype(int)
    preds = pd.Series(np.zeros(len(current)))
    dm.record(current, preds)
    dm.compare()

    features, version = feature_selector.load_feature_set(feature_file)
    assert version is not None and version != base_version
    assert "f2" in features and "f1" not in features

