"""Utilities for lightweight model quantization and pruning."""
from __future__ import annotations

from typing import Optional

import copy

try:  # pragma: no cover - torch optional during import
    import torch
    from torch import nn
    from torch.nn.utils import prune
except Exception:  # pragma: no cover - torch may be absent
    torch = None  # type: ignore
    nn = None  # type: ignore
    prune = None  # type: ignore


def _quantize_tensor(t: torch.Tensor, bits: int) -> torch.Tensor:
    """Quantize a tensor using uniform affine quantization."""
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    scale = t.abs().max()
    if scale == 0:
        return t
    scale = scale / float(qmax)
    q = torch.round(t / scale).clamp(qmin, qmax)
    return q * scale


def apply_quantization(
    model: nn.Module,
    bits: int = 8,
    prune_ratio: float = 0.0,
) -> nn.Module:
    """Return a quantized (and optionally pruned) copy of ``model``.

    Parameters
    ----------
    model:
        The PyTorch module to quantize.
    bits:
        Number of bits for weights. Currently supports any ``bits`` up to 8 via
        simple uniform quantisation. Defaults to 8.
    prune_ratio:
        Optional fraction of channels to prune using ``ln_structured``
        pruning on linear layers. Set to ``0`` to disable.
    """

    if torch is None:
        raise ImportError("torch is required for quantization")

    quantized = copy.deepcopy(model).eval()

    if prune_ratio > 0 and prune is not None:  # pragma: no branch - tiny loop
        for module in quantized.modules():
            if isinstance(module, nn.Linear):
                prune.ln_structured(module, name="weight", amount=prune_ratio, n=2, dim=0)
                prune.remove(module, "weight")

    for name, param in quantized.state_dict().items():
        if param.dtype in (torch.float16, torch.float32, torch.float64):
            param.copy_(_quantize_tensor(param, bits))
    return quantized


__all__ = ["apply_quantization"]
