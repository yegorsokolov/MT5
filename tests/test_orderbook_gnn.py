import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from models.orderbook_gnn import OrderBookGNN, build_orderbook_graph
from rl.trading_env import TradingEnv


def _actual_execution_price(row, size: float = 1.5, depth: int = 2) -> float:
    remaining = size
    cost = 0.0
    for lvl in range(depth):
        price = getattr(row, f"ask_px_{lvl}")
        avail = getattr(row, f"ask_sz_{lvl}")
        take = min(remaining, avail)
        cost += take * price
        remaining -= take
        if remaining <= 0:
            break
    return cost / size


def test_orderbook_gnn_improves_execution_accuracy():
    # synthetic order books for single symbol
    rng = np.random.default_rng(0)
    rows = []
    for t in range(120):
        mid = 100 + t * 0.01
        bid_px_0 = mid - 0.01
        bid_px_1 = mid - 0.02
        ask_px_0 = mid + 0.01
        ask_px_1 = mid + 0.02
        bid_sz_0 = rng.uniform(1, 2)
        bid_sz_1 = rng.uniform(1, 2)
        ask_sz_0 = rng.uniform(0.5, 1.0)
        ask_sz_1 = rng.uniform(0.5, 1.0)
        rows.append(
            {
                "Timestamp": t,
                "Symbol": "XYZ",
                "mid": mid,
                "bid_px_0": bid_px_0,
                "bid_sz_0": bid_sz_0,
                "bid_px_1": bid_px_1,
                "bid_sz_1": bid_sz_1,
                "ask_px_0": ask_px_0,
                "ask_sz_0": ask_sz_0,
                "ask_px_1": ask_px_1,
                "ask_sz_1": ask_sz_1,
            }
        )
    df = pd.DataFrame(rows)

    # Baseline features: top-of-book only
    X_flat = df[["ask_px_0", "ask_sz_0"]].to_numpy()

    # GNN features using both levels
    gnn = OrderBookGNN(in_channels=3, hidden_channels=8)
    feats = []
    for row in df.itertuples(index=False):
        bids = torch.tensor(
            [[row.bid_px_0, row.bid_sz_0], [row.bid_px_1, row.bid_sz_1]],
            dtype=torch.float32,
        )
        asks = torch.tensor(
            [[row.ask_px_0, row.ask_sz_0], [row.ask_px_1, row.ask_sz_1]],
            dtype=torch.float32,
        )
        x, edge_index = build_orderbook_graph(bids, asks)
        with torch.no_grad():
            feats.append(gnn(x, edge_index).numpy())
    X_gnn = np.vstack(feats)

    y = df.apply(_actual_execution_price, axis=1).to_numpy()
    split = 80
    w_flat, *_ = np.linalg.lstsq(X_flat[:split], y[:split], rcond=None)
    w_gnn, *_ = np.linalg.lstsq(X_gnn[:split], y[:split], rcond=None)
    pred_flat = X_flat[split:] @ w_flat
    pred_gnn = X_gnn[split:] @ w_gnn
    mse_flat = np.mean((pred_flat - y[split:]) ** 2)
    mse_gnn = np.mean((pred_gnn - y[split:]) ** 2)
    assert mse_gnn < mse_flat


def test_trading_env_includes_gnn_obs():
    df = pd.DataFrame(
        {
            "Timestamp": [0, 1],
            "Symbol": ["XYZ", "XYZ"],
            "mid": [100.0, 100.0],
            "bid_px_0": [99.0, 99.0],
            "bid_sz_0": [1.0, 1.0],
            "bid_px_1": [98.0, 98.0],
            "bid_sz_1": [1.0, 1.0],
            "ask_px_0": [101.0, 101.0],
            "ask_sz_0": [1.0, 1.0],
            "ask_px_1": [102.0, 102.0],
            "ask_sz_1": [1.0, 1.0],
        }
    )
    env = TradingEnv(df, features=[], orderbook_depth=2, use_orderbook_gnn=True)
    obs = env.reset()
    # only embedding should remain in observation
    assert obs.shape[0] == env.embedding_dim * env.n_symbols
