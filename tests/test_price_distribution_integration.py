import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "train_price_distribution", ROOT / "train_price_distribution.py"
)
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)  # type: ignore
train_price_distribution = train_mod.train_price_distribution


def test_price_distribution_training_and_percentile():
    np.random.seed(0)
    X = np.random.randn(1000, 2)
    y = 0.3 * X[:, 0] + np.random.normal(scale=0.1, size=1000)
    idx = np.arange(len(X))
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    split = int(len(X) * 0.8)
    train_idx, val_idx = idx[:split], idx[split:]
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    model, metrics = train_price_distribution(
        X_tr,
        y_tr,
        X_val,
        y_val,
        n_components=2,
        epochs=200,
    )
    assert 0.5 < metrics["coverage"] < 1.0
    q = model.percentile(X_val[:1], 0.05, n_samples=2000)[0]
    assert np.isfinite(q)
