"""Graph attention network modules implemented with PyTorch only."""

from __future__ import annotations

from typing import Optional
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
import torch.nn.functional as F


class GATLayer(torch.nn.Module):
    """Basic multi-head graph attention layer.

    This implementation avoids external graph libraries and operates on
    ``edge_index`` tensors in COO format with shape ``[2, num_edges]``.  The
    attention coefficients are normalised across incoming edges for each target
    node.

    Parameters
    ----------
    in_channels: int
        Size of each input node feature.
    out_channels: int
        Size of each output node feature per head.
    heads: int, optional
        Number of attention heads. Defaults to ``1``.
    dropout: float, optional
        Dropout applied to the normalised attention weights.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.lin = torch.nn.Linear(in_channels, heads * out_channels, bias=False)
        self.attn = torch.nn.Parameter(torch.empty(heads, 2 * out_channels))
        self.dropout = torch.nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.attn)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.lin(x).view(-1, self.heads, self.out_channels)
        row, col = edge_index
        h_row = h[row]
        h_col = h[col]
        alpha = torch.cat([h_row, h_col], dim=-1)
        alpha = (alpha * self.attn).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha_norm = torch.zeros_like(alpha)
        for i in range(x.size(0)):
            mask = row == i
            if mask.any():
                alpha_norm[mask] = torch.softmax(alpha[mask], dim=0)
        alpha = self.dropout(alpha_norm)
        out = torch.zeros(x.size(0), self.heads, self.out_channels, device=x.device)
        out.index_add_(0, row, alpha.unsqueeze(-1) * h_col)
        return out.mean(dim=1), alpha


class GATNet(torch.nn.Module):
    """Simple multi-layer graph attention network.

    Parameters
    ----------
    in_channels: int
        Size of input node features.
    hidden_channels: int, optional
        Hidden feature size.
    out_channels: int, optional
        Output feature size.
    num_layers: int, optional
        Number of layers in the network. Must be ``>= 2``.
    heads: int, optional
        Number of attention heads per layer.
    dropout: float, optional
        Dropout applied to attention coefficients.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        out_channels: int = 1,
        num_layers: int = 2,
        heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        self.layers = torch.nn.ModuleList()
        self.layers.append(GATLayer(in_channels, hidden_channels, heads, dropout))
        for _ in range(num_layers - 2):
            self.layers.append(GATLayer(hidden_channels, hidden_channels, heads, dropout))
        self.layers.append(GATLayer(hidden_channels, out_channels, heads, dropout))
        self.last_attention: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x, _ = layer(x, edge_index)
            x = F.relu(x)
        x, attn = self.layers[-1](x, edge_index)
        self.last_attention = attn.detach()
        self.last_edge_index = edge_index.detach()  # type: ignore[attr-defined]
        return x


__all__ = ["GATLayer", "GATNet"]
