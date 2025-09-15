import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from train_graphnet import build_graph_examples, train_graphnet


def _make_dataset(n_steps: int = 40, n_symbols: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(n_steps + 1, n_symbols)).astype(np.float32)
    for t in range(n_steps):
        base[t + 1] = base[t] + 0.5 * np.roll(base[t], 1)
    rows = []
    matrices: list[tuple[pd.Timestamp, np.ndarray]] = []
    symbols = [f"S{i}" for i in range(n_symbols)]
    for t in range(n_steps):
        corr = np.eye(n_symbols, dtype=np.float32) * 0.0
        for i in range(n_symbols):
            j = (i + 1) % n_symbols
            corr[i, j] = corr[j, i] = 0.8
        matrices.append((pd.Timestamp(t), corr))
        for idx, sym in enumerate(symbols):
            rows.append(
                {
                    "Timestamp": pd.Timestamp(t),
                    "Symbol": sym,
                    "return": float(base[t, idx]),
                    "lag_return": float(base[t - 1, idx]) if t > 0 else 0.0,
                }
            )
    df = pd.DataFrame(rows)
    df.attrs["adjacency_matrices"] = matrices
    return df


def test_graphnet_converges():
    df = _make_dataset()
    cfg = {
        "symbols": sorted(df["Symbol"].unique()),
        "graph": {"epochs": 120, "lr": 0.01, "hidden_channels": 16},
    }
    model, losses = train_graphnet(df, cfg, return_losses=True)
    assert losses[0] > losses[-1] * 5

    examples = build_graph_examples(df, cfg["symbols"], ["return", "lag_return"])
    first = examples[0]
    pred = model(first.x, first.edge_index, first.edge_weight)
    assert pred.shape == (len(cfg["symbols"]), 1)
