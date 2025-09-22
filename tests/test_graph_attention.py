import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from mt5.train_graphnet import train_graphnet


def _make_dataset(n_steps: int = 30, n_symbols: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    base = rng.normal(size=(n_steps + 1, n_symbols)).astype(np.float32)
    for t in range(n_steps):
        base[t + 1] = base[t] + 0.3 * np.roll(base[t], 1)
    rows = []
    matrices: list[tuple[pd.Timestamp, np.ndarray]] = []
    symbols = [f"S{i}" for i in range(n_symbols)]
    for t in range(n_steps):
        corr = np.zeros((n_symbols, n_symbols), dtype=np.float32)
        for i in range(n_symbols):
            j = (i + 1) % n_symbols
            corr[i, j] = corr[j, i] = 0.9
        matrices.append((pd.Timestamp(t), corr))
        for idx, sym in enumerate(symbols):
            rows.append(
                {
                    "Timestamp": pd.Timestamp(t),
                    "Symbol": sym,
                    "return": float(base[t, idx]),
                }
            )
    df = pd.DataFrame(rows)
    df.attrs["adjacency_matrices"] = matrices
    return df


def test_gat_trains_and_attention_sums(caplog):
    df = _make_dataset()
    cfg = {
        "symbols": sorted(df["Symbol"].unique()),
        "graph": {
            "epochs": 160,
            "lr": 0.01,
            "use_gat": True,
            "heads": 2,
            "dropout": 0.0,
            "log_attention": True,
            "log_top_k_edges": 2,
        },
    }
    with caplog.at_level("INFO"):
        model, losses = train_graphnet(df, cfg, return_losses=True)
    assert losses[0] > losses[-1] * 5
    assert model.last_attention is not None
    assert any("top attention weights" in rec.message for rec in caplog.records)

    edge_index = model.last_edge_index
    attn = model.last_attention
    sums = torch.zeros(len(cfg["symbols"]), attn.size(1))
    sums.index_add_(0, edge_index[0], attn)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
