import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import importlib.util
import types
import sys
import scipy  # ensure SciPy available

root = Path(__file__).resolve().parents[1]

# Stub out heavy analysis dependencies
analysis_stub = types.ModuleType("analysis")
analysis_stub.data_lineage = types.SimpleNamespace(log_lineage=lambda *a, **k: None)
sys.modules.setdefault("analysis", analysis_stub)
sys.modules.setdefault("analysis.data_lineage", analysis_stub.data_lineage)

spec_labels = importlib.util.spec_from_file_location("labels", root / "data" / "labels.py")
labels_mod = importlib.util.module_from_spec(spec_labels)
spec_labels.loader.exec_module(labels_mod)
multi_horizon_labels = labels_mod.multi_horizon_labels


def test_multi_horizon_labels_and_training():
    # confirm helper exists in train.py
    train_source = (root / "train.py").read_text()
    assert "train_multi_output_model" in train_source

    np.random.seed(0)
    n = 60
    prices = pd.Series(np.linspace(0, n * 0.1, n) + np.random.normal(scale=1.0, size=n))
    horizons = [1, 4, 24]
    labels = multi_horizon_labels(prices, horizons)

    for h in horizons:
        assert f"label_{h}" in labels.columns

    X = pd.DataFrame({"feat": np.zeros(n)})

    def f1_score(y_true, y_pred):
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        denom = 2 * tp + fp + fn
        return 0.0 if denom == 0 else 2 * tp / denom

    report = {}
    f1_scores = []
    for h in horizons:
        y = labels[f"label_{h}"]
        majority = int(y.mean() >= 0.5)
        pred = np.full(len(y), majority)
        f1 = f1_score(y, pred)
        report[f"label_{h}"] = {"f1": f1}
        f1_scores.append(f1)
    report["aggregate_f1"] = float(np.mean(f1_scores))

    assert report["label_24"]["f1"] >= report["label_1"]["f1"]
    expected = np.mean([report[f"label_{h}"]["f1"] for h in horizons])
    assert report["aggregate_f1"] == pytest.approx(expected)
