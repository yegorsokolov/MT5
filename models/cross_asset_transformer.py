"""Cross-asset transformer encoder.

This module provides a compact transformer that jointly attends across
multiple trading instruments and time steps.  The forward pass expects
input of shape ``(batch, symbols, seq_len, features)`` and produces a
per-symbol output of shape ``(batch, symbols, output_dim)``.  It is a
lightweight building block intended for environments where a small joint
model over several assets is required.
"""

from __future__ import annotations

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class CrossAssetTransformer(nn.Module):
    """Transformer encoder attending across instruments and time."""

    def __init__(
        self,
        input_dim: int,
        n_symbols: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.n_symbols = n_symbols
        self.input_proj = nn.Linear(input_dim, d_model)
        self.symbol_emb = nn.Embedding(n_symbols, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-symbol outputs for ``x``.

        Parameters
        ----------
        x:
            Tensor with shape ``(batch, symbols, seq_len, features)``.
        """

        b, s, t, _ = x.shape
        x = self.input_proj(x)
        sym_idx = torch.arange(s, device=x.device).view(1, s, 1).expand(b, s, t)
        x = x + self.symbol_emb(sym_idx)
        x = x.reshape(b, s * t, -1)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.reshape(b, s, t, -1)
        return self.head(x[:, :, -1, :])


__all__ = ["CrossAssetTransformer"]
