from __future__ import annotations

"""Cross-modal transformer combining price windows with news embeddings."""

import logging
import math
from typing import Dict, List, Tuple

import torch
from torch import nn

from .time_encoding import TimeEncoding


logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.d_model = d_model
        pe = self._build_encoding(max_len, device=None)
        self.register_buffer("pe", pe, persistent=False)

    def _build_encoding(self, length: int, device: torch.device | None) -> torch.Tensor:
        position = torch.arange(0, length, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(length, self.d_model, dtype=torch.float32, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            pe = self._build_encoding(x.size(1), device=x.device)
            self.register_buffer("pe", pe, persistent=False)
        return x + self.pe[:, : x.size(1)].to(x.device)


class FeedForward(nn.Module):
    """Position-wise feed-forward network used inside transformer layers."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModalEncoder(nn.Module):
    """Self-attention encoder applied independently to each modality."""

    def __init__(self, d_model: int, nhead: int, dropout: float, ff_dim: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class CrossAttentionBlock(nn.Module):
    """Layer containing self-attention per branch and bi-directional cross attention."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        ff_dim: int,
        *,
        average_attn_weights: bool = True,
    ) -> None:
        super().__init__()
        self.price_encoder = ModalEncoder(d_model, nhead, dropout, ff_dim)
        self.news_encoder = ModalEncoder(d_model, nhead, dropout, ff_dim)
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
        self.average_attn_weights = average_attn_weights

    def forward(
        self, price: torch.Tensor, news: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        price_encoded = self.price_encoder(price)
        news_encoded = self.news_encoder(news)
        price_ctx, price_weights = self.price_cross(
            price_encoded,
            news_encoded,
            news_encoded,
            need_weights=True,
            average_attn_weights=self.average_attn_weights,
        )
        price_out = self.price_norm(price_encoded + self.dropout(price_ctx))
        price_out = self.price_ff_norm(price_out + self.dropout(self.price_ff(price_out)))
        news_ctx, news_weights = self.news_cross(
            news_encoded,
            price_out,
            price_out,
            need_weights=True,
            average_attn_weights=self.average_attn_weights,
        )
        news_out = self.news_norm(news_encoded + self.dropout(news_ctx))
        news_out = self.news_ff_norm(news_out + self.dropout(self.news_ff(news_out)))
        return price_out, news_out, price_weights, news_weights


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
        *,
        average_attn_weights: bool = True,
    ) -> None:
        super().__init__()
        if price_dim <= 0 or news_dim <= 0:
            raise ValueError("price_dim and news_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.price_proj = nn.Linear(price_dim, d_model)
        self.news_proj = nn.Linear(news_dim, d_model)
        self.price_pos = PositionalEncoding(d_model)
        self.news_pos = PositionalEncoding(d_model)
        self.price_time_encoder = TimeEncoding(d_model) if time_encoding else None
        self.news_time_encoder = TimeEncoding(d_model) if time_encoding else None
        ff_dim = 4 * d_model
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model,
                    nhead,
                    dropout,
                    ff_dim,
                    average_attn_weights=average_attn_weights,
                )
                for _ in range(num_layers)
            ]
        )
        self.fusion_norm = nn.LayerNorm(2 * d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )
        self.num_layers = num_layers
        self.last_attention: Dict[str, torch.Tensor] | None = None

    def _apply_time_encoding(
        self,
        embedding: torch.Tensor,
        encoder: TimeEncoding | None,
        times: torch.Tensor | None,
    ) -> torch.Tensor:
        if encoder is None or times is None:
            return embedding
        return embedding + encoder(times).type_as(embedding)

    def forward(
        self,
        price: torch.Tensor,
        news: torch.Tensor,
        *,
        price_times: torch.Tensor | None = None,
        news_times: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return probabilities for combined price and news inputs."""

        price_emb = self.price_proj(price)
        news_emb = self.news_proj(news)
        price_emb = self.price_pos(price_emb)
        news_emb = self.news_pos(news_emb)
        price_emb = self._apply_time_encoding(price_emb, self.price_time_encoder, price_times)
        news_emb = self._apply_time_encoding(news_emb, self.news_time_encoder, news_times)

        price_attn: List[torch.Tensor] = []
        news_attn: List[torch.Tensor] = []
        for layer in self.layers:
            price_emb, news_emb, price_weights, news_weights = layer(price_emb, news_emb)
            price_attn.append(price_weights.detach())
            news_attn.append(news_weights.detach())
        if price_attn and news_attn:
            try:
                self.last_attention = {
                    "price_to_news": torch.stack(price_attn),
                    "news_to_price": torch.stack(news_attn),
                }
            except RuntimeError:  # pragma: no cover - inconsistent shapes
                logger.warning("Failed to stack attention maps; storing last layer only")
                self.last_attention = {
                    "price_to_news": price_attn[-1],
                    "news_to_price": news_attn[-1],
                }
        else:
            self.last_attention = None

        price_repr = price_emb.mean(dim=1)
        news_repr = news_emb.mean(dim=1)
        fused = torch.cat([price_repr, news_repr], dim=-1)
        fused = self.fusion_norm(fused)
        logits = self.head(fused)
        if logits.ndim == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        return torch.sigmoid(logits)


__all__ = ["CrossModalTransformer"]
