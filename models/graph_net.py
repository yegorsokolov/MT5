"""Graph neural network models for symbol relation graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

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

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        row, col = edge_index
        agg = torch.zeros_like(x)
        if edge_weight is None:
            agg.index_add_(0, row, x[col])
            deg = torch.bincount(row, minlength=x.size(0)).clamp(min=1).unsqueeze(-1)
        else:
            if edge_weight.dim() == 1:
                ew = edge_weight.unsqueeze(-1)
            else:
                ew = edge_weight
            agg.index_add_(0, row, x[col] * ew)
            deg = torch.zeros(x.size(0), 1, device=x.device)
            deg.index_add_(0, row, ew)
            deg = deg.clamp(min=1)
        agg = agg / deg
        out = torch.cat([x, agg], dim=-1)
        return self.linear(out)


class GraphNet(torch.nn.Module):
    """Simple multi-layer GraphSAGE network.

    The network optionally supports heterogeneous node types by learning a
    small embedding for each type which is concatenated to the input features.

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
    num_node_types: int, optional
        If > 0, the number of distinct node types for which embeddings will be
        learned.  Set ``type_embed_dim`` accordingly.
    type_embed_dim: int, optional
        Dimensionality of the node type embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        out_channels: int = 1,
        num_layers: int = 2,
        num_node_types: int = 0,
        type_embed_dim: int = 0,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.has_type_emb = num_node_types > 0 and type_embed_dim > 0
        if self.has_type_emb:
            self.type_emb = torch.nn.Embedding(num_node_types, type_embed_dim)
            in_channels = in_channels + type_embed_dim

        self.convs = torch.nn.ModuleList()
        self.convs.append(GraphSAGELayer(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GraphSAGELayer(hidden_channels, hidden_channels))
        self.convs.append(GraphSAGELayer(hidden_channels, out_channels))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        node_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.has_type_emb and node_type is not None:
            x = torch.cat([x, self.type_emb(node_type)], dim=-1)
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index, edge_weight))
        return self.convs[-1](x, edge_index, edge_weight)


__all__ = ["GraphNet", "GraphSAGELayer"]
