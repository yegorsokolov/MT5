import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from train_graphnet import train_graphnet


def _make_dataset(n_steps: int = 40, n_symbols: int = 3) -> pd.DataFrame:
    np.random.seed(0)
    base = np.random.randn(n_steps + 1, n_symbols)
    for t in range(n_steps):
        base[t + 1] = base[t] + 0.5 * np.roll(base[t], 1)
    df = pd.DataFrame(base[:-1], columns=[f"S{i}" for i in range(n_symbols)])
    adj = np.zeros((n_symbols, n_symbols))
    for i in range(n_symbols):
        j = (i + 1) % n_symbols
        adj[i, j] = adj[j, i] = 1
    df.attrs["adjacency_matrices"] = [adj for _ in range(len(df))]
    return df


def test_graphnet_converges():
    df = _make_dataset()
    model, losses = train_graphnet(
        df, {"symbols": df.columns, "epochs": 200, "lr": 0.1}, return_losses=True
    )
    assert losses[0] > losses[-1] * 10
    # sanity check a prediction
    x = torch.tensor(df.iloc[0].values, dtype=torch.float32).view(-1, 1)
    edge_index = torch.tensor(
        np.array(df.attrs["adjacency_matrices"][0]).nonzero(), dtype=torch.long
    )
    pred = model(x, edge_index)
    assert pred.shape == (len(df.columns), 1)
