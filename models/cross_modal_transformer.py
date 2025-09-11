from __future__ import annotations

"""Simple cross-modal transformer for price and news inputs."""

import logging
from typing import Dict

import torch
from torch import nn


logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class CrossModalTransformer(nn.Module):
    """Tiny transformer using cross-attention between price and news inputs."""

    def __init__(
        self,
        price_dim: int,
        news_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.price_proj = nn.Linear(price_dim, d_model)
        self.news_proj = nn.Linear(news_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.price_norm = nn.LayerNorm(d_model)
        self.news_norm = nn.LayerNorm(d_model)
        self.p_to_n = nn.ModuleList(
            [nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True) for _ in range(num_layers)]
        )
        self.n_to_p = nn.ModuleList(
            [nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True) for _ in range(num_layers)]
        )
        self.head = nn.Linear(d_model, output_dim)
        self.last_attention: Dict[str, torch.Tensor] | None = None

    def forward(self, price: torch.Tensor, news: torch.Tensor) -> torch.Tensor:
        """Return logits for ``price`` and ``news`` inputs.

        Parameters
        ----------
        price: torch.Tensor
            Tensor of shape ``(batch, seq_len, price_dim)``.
        news: torch.Tensor
            Tensor of shape ``(batch, n_news, news_dim)``.
        """

        p = self.pos_enc(self.price_proj(price))
        n = self.news_proj(news)
        attn: Dict[str, torch.Tensor] = {}
        for attn_p, attn_n in zip(self.p_to_n, self.n_to_p):
            p2, w_pn = attn_p(p, n, n)
            n2, w_np = attn_n(n, p, p)
            p = self.price_norm(p + p2)
            n = self.news_norm(n + n2)
            attn = {"price_to_news": w_pn.detach(), "news_to_price": w_np.detach()}
        self.last_attention = attn
        logger.debug("price->news attention: %s", attn.get("price_to_news"))
        logger.debug("news->price attention: %s", attn.get("news_to_price"))
        out = self.head(p[:, -1, :])
        return torch.sigmoid(out.squeeze(-1))


__all__ = ["CrossModalTransformer"]
