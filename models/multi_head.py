import math
from typing import Dict, Iterable, List

import torch

from .time_encoding import TimeEncoding


class PositionalEncoding(torch.nn.Module):
    """Sinusoidal positional encoding used by the multi-head model."""

    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiTaskHead(torch.nn.Module):
    """Horizon-aware multi-task head.

    A separate linear layer is created for each prediction horizon and task
    (direction, absolute return and volatility).
    """

    def __init__(self, d_model: int, horizons: List[int]) -> None:
        super().__init__()
        self.horizons = [int(h) for h in horizons]
        self.dir = torch.nn.ModuleDict({str(h): torch.nn.Linear(d_model, 1) for h in self.horizons})
        self.abs = torch.nn.ModuleDict({str(h): torch.nn.Linear(d_model, 1) for h in self.horizons})
        self.vol = torch.nn.ModuleDict({str(h): torch.nn.Linear(d_model, 1) for h in self.horizons})

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for h in self.horizons:
            key = str(h)
            out[f"direction_{h}"] = torch.sigmoid(self.dir[key](x)).squeeze(-1)
            out[f"abs_return_{h}"] = self.abs[key](x).squeeze(-1)
            out[f"volatility_{h}"] = self.vol[key](x).squeeze(-1)
        return out


class MultiHeadTransformer(torch.nn.Module):
    """Shared backbone with symbol-specific multi-task heads."""

    def __init__(
        self,
        input_size: int,
        num_symbols: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_regimes: int | None = None,
        emb_dim: int = 8,
        use_checkpointing: bool = False,
        dropout: float = 0.1,
        ff_dim: int | None = None,
        layer_norm: bool = False,
        time_encoding: bool = False,
        horizons: List[int] | None = None,
    ) -> None:
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.horizons = horizons or [1]
        self.regime_emb = None
        self.regime_idx = None
        self.time_encoder = TimeEncoding(d_model) if time_encoding else None
        if num_regimes is not None:
            self.regime_idx = input_size - 1
            self.regime_emb = torch.nn.Embedding(num_regimes, emb_dim)
            input_size -= 1
        emb_total = emb_dim if self.regime_emb is not None else 0
        self.input_linear = torch.nn.Linear(input_size + emb_total, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=ff_dim or 4 * d_model,
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = torch.nn.LayerNorm(d_model) if layer_norm else None
        self.heads = torch.nn.ModuleDict(
            {str(i): MultiTaskHead(d_model, self.horizons) for i in range(num_symbols)}
        )

    def forward(
        self, x: torch.Tensor, symbol: int, times: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.regime_emb is not None and self.regime_idx is not None:
            reg = x[:, :, self.regime_idx].long()
            base = x[:, :, : self.regime_idx]
            x = torch.cat([base, self.regime_emb(reg)], dim=-1)
        x = self.input_linear(x)
        if self.time_encoder is not None:
            if times is None:
                raise ValueError("times tensor required when time_encoding is enabled")
            x = x + self.time_encoder(times)
        x = self.pos_encoder(x)
        if self.use_checkpointing:
            for layer in self.transformer.layers:
                x = torch.utils.checkpoint.checkpoint(layer, x)
        else:
            x = self.transformer(x)
        if self.norm is not None:
            x = self.norm(x)
        head = self.heads[str(int(symbol))]
        return head(x[:, -1])

    def prune_heads(self, keep: Iterable[int]) -> None:
        """Remove heads not listed in ``keep`` to save memory."""
        keep_set = {int(k) for k in keep}
        for k in list(self.heads.keys()):
            if int(k) not in keep_set:
                del self.heads[k]


__all__: List[str] = ["MultiHeadTransformer", "MultiTaskHead"]
