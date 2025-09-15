from __future__ import annotations

"""Cross-modal transformer combining price windows with news embeddings."""

import logging
from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .time_encoding import TimeEncoding


logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class FeedForward(nn.Module):
    """Position-wise feed-forward network used inside transformer layers."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.lin1 = nn.Linear(d_model, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.dropout(F.gelu(self.lin1(x))))


class TransformerBranch(nn.Module):
    """Self-attention block applied independently per modality."""

    def __init__(self, d_model: int, nhead: int, dropout: float, ff_dim: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = FeedForward(d_model, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class CrossModalLayer(nn.Module):
    """Layer containing self-attention per branch and bi-directional cross attention."""

    def __init__(self, d_model: int, nhead: int, dropout: float, ff_dim: int) -> None:
        super().__init__()
        self.price_branch = TransformerBranch(d_model, nhead, dropout, ff_dim)
        self.news_branch = TransformerBranch(d_model, nhead, dropout, ff_dim)
        self.price_cross = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.news_cross = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.price_norm = nn.LayerNorm(d_model)
        self.news_norm = nn.LayerNorm(d_model)
        self.price_ff = FeedForward(d_model, ff_dim, dropout)
        self.news_ff = FeedForward(d_model, ff_dim, dropout)
        self.price_ff_norm = nn.LayerNorm(d_model)
        self.news_ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, price: torch.Tensor, news: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        price = self.price_branch(price)
        news = self.news_branch(news)
        price_ctx, price_weights = self.price_cross(price, news, news, need_weights=True)
        price = self.price_norm(price + self.dropout(price_ctx))
        price = self.price_ff_norm(price + self.dropout(self.price_ff(price)))
        news_ctx, news_weights = self.news_cross(news, price, price, need_weights=True)
        news = self.news_norm(news + self.dropout(news_ctx))
        news = self.news_ff_norm(news + self.dropout(self.news_ff(news)))
        return price, news, price_weights, news_weights


class CrossModalTransformer(nn.Module):
    """Transformer that fuses price windows with news embeddings via cross-attention."""

    def __init__(
        self,
        price_dim: int,
        news_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 1,
        time_encoding: bool = False,
    ) -> None:
        super().__init__()
        self.price_proj = nn.Linear(price_dim, d_model)
        self.news_proj = nn.Linear(news_dim, d_model)
        self.price_pos = PositionalEncoding(d_model)
        self.news_pos = PositionalEncoding(d_model)
        self.price_time_encoder = TimeEncoding(d_model) if time_encoding else None
        self.news_time_encoder = TimeEncoding(d_model) if time_encoding else None
        ff_dim = 4 * d_model
        self.layers = nn.ModuleList(
            [CrossModalLayer(d_model, nhead, dropout, ff_dim) for _ in range(num_layers)]
        )
        self.fusion_norm = nn.LayerNorm(2 * d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )
        self.last_attention: Dict[str, torch.Tensor] | None = None

    def forward(
        self,
        price: torch.Tensor,
        news: torch.Tensor,
        *,
        price_times: torch.Tensor | None = None,
        news_times: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return probabilities for combined price and news inputs.

        Parameters
        ----------
        price_times:
            Optional tensor of timestamps for the price branch. Required when
            ``time_encoding`` was enabled at construction time.
        news_times:
            Optional tensor of timestamps for the news branch. If omitted the
            news stream falls back to standard positional encodings.
        """

        price_emb = self.price_pos(self.price_proj(price))
        news_emb = self.news_pos(self.news_proj(news))
        price_time_features: torch.Tensor | None = None
        news_time_features: torch.Tensor | None = None
        if self.price_time_encoder is not None:
            if price_times is None:
                raise ValueError("price_times required when time encoding is enabled")
            price_time_features = self.price_time_encoder(price_times).type_as(price_emb)
        if self.news_time_encoder is not None and news_times is not None:
            news_time_features = self.news_time_encoder(news_times).type_as(news_emb)
        attn: Dict[str, torch.Tensor] = {}
        for layer in self.layers:
            price_input = price_emb
            news_input = news_emb
            if price_time_features is not None:
                price_input = price_input + price_time_features
            if news_time_features is not None:
                news_input = news_input + news_time_features
            price_emb, news_emb, w_price, w_news = layer(price_input, news_input)
            attn = {"price_to_news": w_price.detach(), "news_to_price": w_news.detach()}
        self.last_attention = attn
        price_repr = price_emb.mean(dim=1)
        news_repr = news_emb.mean(dim=1)
        fused = torch.cat([price_repr, news_repr], dim=-1)
        fused = self.fusion_norm(fused)
        logits = self.head(fused).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs


__all__ = ["CrossModalTransformer"]

