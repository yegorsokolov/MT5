import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from train_graphnet import train_graphnet


def _make_dataset(n_steps: int = 30, n_symbols: int = 3) -> pd.DataFrame:
    np.random.seed(1)
    base = np.random.randn(n_steps + 1, n_symbols)
    for t in range(n_steps):
        base[t + 1] = base[t] + 0.3 * np.roll(base[t], 1)
    df = pd.DataFrame(base[:-1], columns=[f"S{i}" for i in range(n_symbols)])
    adj = np.zeros((n_symbols, n_symbols))
    for i in range(n_symbols):
        j = (i + 1) % n_symbols
        adj[i, j] = adj[j, i] = 1
    df.attrs["adjacency_matrices"] = [adj for _ in range(len(df))]
    return df


def test_gat_trains_and_attention_sums():
    df = _make_dataset()
    model, losses = train_graphnet(
        df,
        {
            "symbols": df.columns,
            "epochs": 300,
            "lr": 0.001,
            "use_gat": True,
            "gat_heads": 2,
            "gat_dropout": 0.0,
        },
        return_losses=True,
    )
    assert losses[0] > losses[-1] * 10
    assert model.last_attention is not None
    edge_index = torch.tensor(
        np.array(df.attrs["adjacency_matrices"][0]).nonzero(), dtype=torch.long
    )
    attn = model.last_attention
    sums = torch.zeros(len(df.columns), attn.size(1))
    sums.index_add_(0, edge_index[0], attn)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
