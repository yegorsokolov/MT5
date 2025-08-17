"""Temporal Fusion Transformer wrapper.

This module provides a lightweight implementation of the core pieces of the
Temporal Fusion Transformer (TFT) architecture [1]_. It supports static,
known and observed inputs and produces multi-quantile forecasts. The
implementation is intentionally simplified so that it can run in constrained
execution environments without relying on external heavy dependencies such as
``pytorch-forecasting``. Only a small subset of the original TFT is
implemented – enough for unit tests and small scale experiments.

The model exposes ``last_attention`` and ``variable_importance`` information for
interpretability. ``last_attention`` stores the attention weights produced on
the most recent forward pass while ``variable_importance`` computes a simple
feature importance measure from the absolute values of the first layer weights.

References
----------
.. [1] Bryan Lim et al., "Temporal Fusion Transformers for Interpretable
   Multi-horizon Time Series Forecasting", 2020.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn


class QuantileLoss(nn.Module):
    """Pinball loss for multiple quantiles.

    Parameters
    ----------
    quantiles: Iterable[float]
        Sequence of quantiles to predict, e.g. ``[0.1, 0.5, 0.9]``.
    """

    def __init__(self, quantiles: Iterable[float]) -> None:
        super().__init__()
        q = torch.tensor(list(quantiles), dtype=torch.float32)
        if q.ndim != 1:
            raise ValueError("quantiles must be a 1D sequence")
        self.register_buffer("quantiles", q.view(1, -1))

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the pinball loss for ``preds`` against ``target``.

        ``preds`` should have shape ``[batch, num_quantiles]`` and ``target``
        either ``[batch]`` or ``[batch, num_quantiles]``. The output is the mean
        loss over batch and quantiles.
        """

        if target.ndim == 1:
            target = target.unsqueeze(-1)
        errors = target - preds
        loss = torch.maximum((self.quantiles - 1) * errors, self.quantiles * errors)
        return loss.mean()


@dataclass
class TFTConfig:
    """Configuration for :class:`TemporalFusionTransformer`."""

    static_size: int = 0
    known_size: int = 0
    observed_size: int = 0
    hidden_size: int = 32
    num_heads: int = 4
    quantiles: Iterable[float] = (0.1, 0.5, 0.9)


class TemporalFusionTransformer(nn.Module):
    """A very small Temporal Fusion Transformer.

    The model is heavily simplified compared to the original TFT but retains the
    key ideas: an LSTM encoder, a self-attention layer and a projection to
    quantile outputs. Static features are broadcast across the time dimension
    and concatenated with known and observed inputs.
    """

    def __init__(self, config: TFTConfig) -> None:
        super().__init__()
        self.config = config
        inp_size = config.static_size + config.known_size + config.observed_size
        self.input_linear = nn.Linear(inp_size, config.hidden_size)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)
        self.attn = nn.MultiheadAttention(
            config.hidden_size, config.num_heads, batch_first=True
        )
        self.proj = nn.Linear(config.hidden_size, len(list(config.quantiles)))
        self.register_buffer(
            "_quantiles", torch.tensor(list(config.quantiles), dtype=torch.float32)
        )
        self.last_attention: torch.Tensor | None = None

    def _combine_inputs(
        self, static: torch.Tensor | None, known: torch.Tensor, observed: torch.Tensor | None
    ) -> torch.Tensor:
        seq_len = known.size(1)
        parts = []
        if static is not None:
            static = static.unsqueeze(1).expand(-1, seq_len, -1)
            parts.append(static)
        parts.append(known)
        if observed is not None:
            parts.append(observed)
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        known: torch.Tensor,
        static: torch.Tensor | None = None,
        observed: torch.Tensor | None = None,
        _code: int | None = None,
    ) -> torch.Tensor:
        """Forward pass returning quantile forecasts."""

        x = self._combine_inputs(static, known, observed)
        x = torch.relu(self.input_linear(x))
        lstm_out, _ = self.lstm(x)
        attn_out, attn_weights = self.attn(lstm_out, lstm_out, lstm_out)
        self.last_attention = attn_weights.detach()
        out = self.proj(attn_out[:, -1, :])
        return out

    def variable_importance(self) -> dict[str, float]:
        """Return a simple variable importance score for each input feature."""

        weights = self.input_linear.weight.abs().sum(0)
        total = weights.sum().item() or 1.0
        names: List[str] = []
        if self.config.static_size:
            names.extend(f"static_{i}" for i in range(self.config.static_size))
        if self.config.known_size:
            names.extend(f"known_{i}" for i in range(self.config.known_size))
        if self.config.observed_size:
            names.extend(f"observed_{i}" for i in range(self.config.observed_size))
        scores = (weights / total).detach().cpu().tolist()
        return dict(zip(names, scores))

    def predict_quantiles(
        self,
        known: torch.Tensor,
        static: torch.Tensor | None = None,
        observed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convenience method for obtaining quantile forecasts in eval mode."""

        self.eval()
        with torch.no_grad():
            return self(known, static=static, observed=observed)


__all__ = ["TemporalFusionTransformer", "TFTConfig", "QuantileLoss"]
