"""Differentiable risk loss functions and utilities."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:  # pragma: no cover - torch is optional
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


def _to_tensor(x):
    arr = np.asarray(x, dtype="float32")
    if _TORCH_AVAILABLE:
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.tensor(arr)
    return arr


def cvar(returns, level: float = 0.05):
    r = _to_tensor(returns)
    if _TORCH_AVAILABLE and isinstance(r, torch.Tensor):
        var = torch.quantile(r, level)
        tail = r[r <= var]
        return tail.mean()
    var = np.quantile(r, level)
    tail = r[r <= var]
    return float(tail.mean()) if tail.size else 0.0


def max_drawdown(returns):
    r = _to_tensor(returns)
    if _TORCH_AVAILABLE and isinstance(r, torch.Tensor):
        cumulative = torch.cumsum(r, dim=0)
        peak = torch.cummax(cumulative, dim=0)[0]
        drawdown = (peak - cumulative) / torch.clamp(peak, min=1e-8)
        return drawdown.max()
    cumulative = np.cumsum(r)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / np.clip(peak, 1e-8, None)
    return float(drawdown.max()) if drawdown.size else 0.0


def cvar_penalty(returns, limit: float, level: float = 0.05, scale: float = 1.0):
    cvar_val = -cvar(returns, level)
    if _TORCH_AVAILABLE and isinstance(cvar_val, torch.Tensor):
        return scale * torch.relu(cvar_val - limit)
    return float(scale * max(cvar_val - limit, 0.0))


def max_drawdown_penalty(returns, limit: float, scale: float = 1.0):
    mdd = max_drawdown(returns)
    if _TORCH_AVAILABLE and isinstance(mdd, torch.Tensor):
        return scale * torch.relu(mdd - limit)
    return float(scale * max(mdd - limit, 0.0))


@dataclass
class RiskBudget:
    """Simple risk budget specifying leverage and drawdown limits."""

    max_leverage: float
    max_drawdown: float
    cvar_limit: float | None = None

    def as_features(self) -> dict[str, float]:
        feats = {
            "risk_max_leverage": float(self.max_leverage),
            "risk_max_drawdown": float(self.max_drawdown),
        }
        if self.cvar_limit is not None:
            feats["risk_cvar_limit"] = float(self.cvar_limit)
        return feats

    def scale_positions(self, positions):
        p = _to_tensor(positions)
        if _TORCH_AVAILABLE and isinstance(p, torch.Tensor):
            lev = p.abs().sum()
            scale = torch.clamp(self.max_leverage / (lev + 1e-8), max=1.0)
            return p * scale
        lev = np.abs(p).sum()
        if lev <= self.max_leverage or lev == 0:
            return p
        return p * (self.max_leverage / lev)


def risk_penalty(returns, budget: RiskBudget, level: float = 0.05, scale: float = 1.0):
    pen = 0.0
    if budget.cvar_limit is not None:
        pen = pen + cvar_penalty(returns, budget.cvar_limit, level, scale)
    pen = pen + max_drawdown_penalty(returns, budget.max_drawdown, scale)
    return pen


__all__ = [
    "cvar",
    "max_drawdown",
    "cvar_penalty",
    "max_drawdown_penalty",
    "risk_penalty",
    "RiskBudget",
]
