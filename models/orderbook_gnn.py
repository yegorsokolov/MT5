"""Order book encoder using a tiny GraphSAGE network."""

from __future__ import annotations

from typing import Tuple

import torch


class GraphSAGELayer(torch.nn.Module):
    """Minimal GraphSAGE layer using mean aggregation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, x[col])
        deg = torch.bincount(row, minlength=x.size(0)).clamp(min=1).unsqueeze(-1)
        agg = agg / deg
        out = torch.cat([x, agg], dim=-1)
        return self.linear(out)


class OrderBookGNN(torch.nn.Module):
    """Encode bid/ask price levels of an order book into a single vector.

    The network is deliberately lightweight so it can run in constrained
    environments.  ``forward`` expects node features ``x`` of shape
    ``[num_levels * 2, in_channels]`` and ``edge_index`` in standard COO
    format with shape ``[2, num_edges]``.  The first half of nodes represent
    bid levels ordered from best to worst, followed by ask levels.
    """

    def __init__(self, in_channels: int = 3, hidden_channels: int = 16) -> None:
        super().__init__()
        self.conv1 = GraphSAGELayer(in_channels, hidden_channels)
        self.conv2 = GraphSAGELayer(hidden_channels, hidden_channels)
        self.hidden_channels = hidden_channels

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        # Mean pool over all nodes to get a single embedding
        return x.mean(dim=0)


def build_orderbook_graph(
    bids: torch.Tensor,
    asks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build node features and edges from bid/ask arrays.

    Parameters
    ----------
    bids, asks: torch.Tensor
        Tensors of shape ``[levels, 2]`` containing price and size for each
        level.  ``bids`` should be sorted from best to worst price, ``asks``
        from best to worst (lowest to highest).
    """

    levels = bids.size(0)
    # Node features: [price, size, side]
    side_bid = torch.full((levels, 1), -1.0)
    side_ask = torch.full((levels, 1), 1.0)
    x = torch.cat([
        torch.cat([bids, side_bid], dim=1),
        torch.cat([asks, side_ask], dim=1),
    ])

    # Build edges connecting consecutive levels on each side and the best levels
    edges = []
    for offset in (0, levels):
        for i in range(levels - 1):
            edges.append((offset + i, offset + i + 1))
            edges.append((offset + i + 1, offset + i))
    # Connect best bid with best ask
    edges.append((0, levels))
    edges.append((levels, 0))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return x, edge_index


__all__ = ["OrderBookGNN", "build_orderbook_graph"]
