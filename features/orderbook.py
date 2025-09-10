"""Order book graph embedding features."""

from __future__ import annotations

from typing import Optional
import logging

import pandas as pd

try:  # Optional dependency for environments without torch
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from models.orderbook_gnn import OrderBookGNN, build_orderbook_graph
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    DataLoader = TensorDataset = OrderBookGNN = build_orderbook_graph = None  # type: ignore

logger = logging.getLogger(__name__)


def _infer_depth(df: pd.DataFrame) -> int:
    depth = 0
    while (
        f"bid_px_{depth}" in df.columns
        and f"bid_sz_{depth}" in df.columns
        and f"ask_px_{depth}" in df.columns
        and f"ask_sz_{depth}" in df.columns
    ):
        depth += 1
    return depth


def compute(
    df: pd.DataFrame,
    depth: Optional[int] = None,
    hidden_channels: int = 16,
    batch_size: int = 512,
) -> pd.DataFrame:
    """Compute order book embeddings using :class:`OrderBookGNN`.

    Parameters
    ----------
    df:
        DataFrame containing bid/ask price and size columns. Columns are expected
        to follow the pattern ``bid_px_<i>``, ``bid_sz_<i>``, ``ask_px_<i>``,
        ``ask_sz_<i>`` where ``i`` ranges from ``0`` to ``depth-1``.
    depth:
        Number of price levels to use. If ``None`` the depth is inferred from the
        columns present.
    hidden_channels:
        Size of the embedding produced by the GNN.
    batch_size:
        Number of samples to process per forward pass.  Larger batches may be
        faster but consume more memory.

    Returns
    -------
    ``pd.DataFrame`` with additional columns ``ob_emb_0`` .. ``ob_emb_{hidden_channels-1}``.
    """

    if OrderBookGNN is None or torch is None:  # pragma: no cover - dependency missing
        logger.debug("OrderBookGNN not available; skipping order book features")
        return df

    df = df.copy()
    if depth is None:
        depth = _infer_depth(df)
    if depth == 0:
        return df

    gnn = OrderBookGNN(in_channels=3, hidden_channels=hidden_channels)

    # Convert the entire DataFrame slice to tensors once to avoid Python-level
    # loops.  Shape: [N, depth, 2]
    bids = torch.stack(
        [
            torch.tensor(
                df[[f"bid_px_{i}", f"bid_sz_{i}"]].values, dtype=torch.float32
            )
            for i in range(depth)
        ],
        dim=1,
    )
    asks = torch.stack(
        [
            torch.tensor(
                df[[f"ask_px_{i}", f"ask_sz_{i}"]].values, dtype=torch.float32
            )
            for i in range(depth)
        ],
        dim=1,
    )

    dataset = TensorDataset(bids, asks)
    loader = DataLoader(dataset, batch_size=batch_size)

    # Base edge index is identical for all samples; construct once.
    dummy = torch.zeros(depth, 2)
    _, edge_index_base = build_orderbook_graph(dummy, dummy)
    num_nodes = 2 * depth
    num_edges = edge_index_base.size(1)

    embeddings = []
    with torch.no_grad():
        for bids_batch, asks_batch in loader:
            bsz = bids_batch.size(0)
            side_bid = torch.full((bsz, depth, 1), -1.0)
            side_ask = torch.full((bsz, depth, 1), 1.0)
            x_batch = torch.cat(
                [
                    torch.cat([bids_batch, side_bid], dim=2),
                    torch.cat([asks_batch, side_ask], dim=2),
                ],
                dim=1,
            )  # [B, 2*depth, 3]
            x_flat = x_batch.view(bsz * num_nodes, 3)
            offsets = (torch.arange(bsz) * num_nodes).repeat_interleave(num_edges)
            edge_index = edge_index_base.repeat(1, bsz) + offsets.unsqueeze(0)
            h = torch.relu(gnn.conv1(x_flat, edge_index))
            h = gnn.conv2(h, edge_index)
            embeddings.append(h.view(bsz, num_nodes, -1).mean(dim=1))

    emb = torch.cat(embeddings, dim=0)
    emb_cols = [f"ob_emb_{i}" for i in range(gnn.hidden_channels)]
    emb_df = pd.DataFrame(emb.numpy(), columns=emb_cols, index=df.index)
    return pd.concat([df, emb_df], axis=1)


__all__ = ["compute"]
