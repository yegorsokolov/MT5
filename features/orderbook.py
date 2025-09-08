"""Order book graph embedding features."""

from __future__ import annotations

from typing import List, Optional
import logging

import pandas as pd

try:  # Optional dependency for environments without torch
    import torch
    from models.orderbook_gnn import OrderBookGNN, build_orderbook_graph
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    OrderBookGNN = build_orderbook_graph = None  # type: ignore

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
    embeddings: List[torch.Tensor] = []

    for row in df.itertuples(index=False):
        bids = torch.tensor(
            [
                [getattr(row, f"bid_px_{i}"), getattr(row, f"bid_sz_{i}")]
                for i in range(depth)
            ],
            dtype=torch.float32,
        )
        asks = torch.tensor(
            [
                [getattr(row, f"ask_px_{i}"), getattr(row, f"ask_sz_{i}")]
                for i in range(depth)
            ],
            dtype=torch.float32,
        )
        x, edge_index = build_orderbook_graph(bids, asks)
        with torch.no_grad():
            embeddings.append(gnn(x, edge_index))

    emb_cols = [f"ob_emb_{i}" for i in range(gnn.hidden_channels)]
    emb_df = pd.DataFrame(
        torch.stack(embeddings).numpy(), columns=emb_cols, index=df.index
    )
    return pd.concat([df, emb_df], axis=1)


__all__ = ["compute"]
