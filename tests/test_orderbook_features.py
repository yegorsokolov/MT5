import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from features.orderbook import compute


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
