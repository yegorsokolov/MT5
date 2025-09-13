"""Differentiable risk objectives for training.

This module provides penalty functions for Conditional Value at Risk (CVaR)
and maximum drawdown that are compatible with both NumPy arrays and PyTorch
``Tensor`` objects.  The functions return scalar penalties that are zero when
the corresponding risk metric is below a user supplied target and grow
linearly once the target is breached.  When PyTorch is available the returned
penalties participate in autograd allowing the objectives to be used during
neural network training.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - torch is optional
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


def _to_tensor(x: np.ndarray | list[float]) -> "torch.Tensor":
    """Convert input to a Torch tensor if possible."""

    arr = np.asarray(x, dtype="float32")
    if _TORCH_AVAILABLE:
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.tensor(arr)

    # Fallback: create a minimal tensor-like object using NumPy
    class _ArrayWrapper(np.ndarray):
        pass

    return arr.view(_ArrayWrapper)


def cvar(returns: np.ndarray | list[float] | "torch.Tensor", level: float = 0.05):
    """Differentiable Conditional Value at Risk.

    Parameters
    ----------
    returns:
        Sequence of portfolio returns.
    level:
        Tail probability level.  A value of ``0.05`` corresponds to the 5%
        worst losses.
    """

    r = _to_tensor(returns)
    if _TORCH_AVAILABLE and isinstance(r, torch.Tensor):
        var = torch.quantile(r, level)
        tail = r[r <= var]
        return tail.mean()
    var = np.quantile(r, level)
    tail = r[r <= var]
    return float(tail.mean()) if tail.size else 0.0


def max_drawdown(returns: np.ndarray | list[float] | "torch.Tensor"):
    """Differentiable maximum drawdown of a return series."""

    r = _to_tensor(returns)
    if _TORCH_AVAILABLE and isinstance(r, torch.Tensor):
        cumulative = torch.cumsum(r, dim=0)
        peak = torch.cummax(cumulative, dim=0)[0]
        drawdown = peak - cumulative
        return drawdown.max()
    cumulative = np.cumsum(r)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    return float(drawdown.max()) if drawdown.size else 0.0


def cvar_penalty(
    returns: np.ndarray | list[float] | "torch.Tensor",
    target: float,
    level: float = 0.05,
    scale: float = 1.0,
):
    """Return penalty for exceeding a CVaR target."""

    cvar_val = -cvar(returns, level)
    if _TORCH_AVAILABLE and isinstance(cvar_val, torch.Tensor):
        return scale * torch.relu(cvar_val - target)
    return float(scale * max(cvar_val - target, 0.0))


def max_drawdown_penalty(
    returns: np.ndarray | list[float] | "torch.Tensor",
    target: float,
    scale: float = 1.0,
):
    """Return penalty for exceeding a maximum drawdown target."""

    mdd = max_drawdown(returns)
    if _TORCH_AVAILABLE and isinstance(mdd, torch.Tensor):
        return scale * torch.relu(mdd - target)
    return float(scale * max(mdd - target, 0.0))


def risk_penalty(
    returns: np.ndarray | list[float] | "torch.Tensor",
    *,
    cvar_target: float | None = None,
    mdd_target: float | None = None,
    level: float = 0.05,
    scale: float = 1.0,
):
    """Aggregate CVaR and drawdown penalties for a return series."""

    penalty = 0.0
    if cvar_target is not None:
        penalty = penalty + cvar_penalty(returns, cvar_target, level, scale)
    if mdd_target is not None:
        penalty = penalty + max_drawdown_penalty(returns, mdd_target, scale)
    return penalty


__all__ = [
    "cvar",
    "max_drawdown",
    "cvar_penalty",
    "max_drawdown_penalty",
    "risk_penalty",
]
