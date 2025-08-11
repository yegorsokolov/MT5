"""Graph neural network models for symbol relation graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


class GraphSAGELayer(torch.nn.Module):
    """Minimal GraphSAGE layer using mean aggregation.

    This implementation avoids external graph libraries so it can run in
    lightweight environments. ``edge_index`` follows the standard COO format
    with shape ``[2, num_edges]``.
    """

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


class GraphNet(torch.nn.Module):
    """Simple multi-layer GraphSAGE network.

    Parameters
    ----------
    in_channels: int
        Size of input node features.
    hidden_channels: int
        Size of hidden representations.
    out_channels: int
        Output size per node (e.g. regression target or class logits).
    num_layers: int, optional
        Number of GraphSAGE layers. ``num_layers`` >= 2.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        out_channels: int = 1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        self.convs = torch.nn.ModuleList()
        self.convs.append(GraphSAGELayer(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GraphSAGELayer(hidden_channels, hidden_channels))
        self.convs.append(GraphSAGELayer(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        return self.convs[-1](x, edge_index)


__all__ = ["GraphNet", "GraphSAGELayer"]
