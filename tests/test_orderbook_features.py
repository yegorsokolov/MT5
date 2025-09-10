import time
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import importlib.util

torch = pytest.importorskip("torch")

spec = importlib.util.spec_from_file_location(
    "orderbook", Path(__file__).resolve().parents[1] / "features" / "orderbook.py"
)
orderbook = importlib.util.module_from_spec(spec)
assert spec.loader is not None  # for mypy
spec.loader.exec_module(orderbook)
compute = orderbook.compute


def _make_df(rows: int = 5) -> pd.DataFrame:
    data = []
    for t in range(rows):
        data.append(
            {
                "Timestamp": t,
                "bid_px_0": 99.0 + t * 0.01,
                "bid_sz_0": 1.0 + 0.1 * t,
                "bid_px_1": 98.0 + t * 0.01,
                "bid_sz_1": 1.5 + 0.1 * t,
                "ask_px_0": 101.0 + t * 0.01,
                "ask_sz_0": 1.0 + 0.1 * t,
                "ask_px_1": 102.0 + t * 0.01,
                "ask_sz_1": 1.2 + 0.1 * t,
            }
        )
    return pd.DataFrame(data)


def test_orderbook_compute_adds_embeddings():
    df = _make_df(10)
    out = compute(df, depth=2, hidden_channels=8)
    cols = [c for c in out.columns if c.startswith("ob_emb_")]
    assert len(cols) == 8
    emb = out[cols].to_numpy()
    assert emb.shape == (len(df), 8)
    assert np.isfinite(emb).all()


def test_orderbook_compute_performance():
    df = _make_df(2000)
    start = time.time()
    compute(df, depth=2, hidden_channels=8, batch_size=256)
    duration = time.time() - start
    # Should comfortably run within a second on moderate datasets
    assert duration < 1.0
