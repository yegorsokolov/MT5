from __future__ import annotations

"""Differentiable backtesting utilities.

The goal of this module is not to provide a production grade simulator but
rather a *smooth* approximation of PnL that allows gradients to flow through
trading decisions.  It is intentionally lightweight so it can be used in unit
tests and small research experiments.

Two key ideas are used:

* Orders are represented by logits for ``buy``, ``sell`` and ``hold``.  A
  softmax over these logits yields a continuous position in ``[-1, 1]`` so the
  entire operation is differentiable.
* Transaction costs are modelled via a differentiable proxy of slippage based on
  the turnover of the position.

These simplifications enable end‑to‑end training of neural strategies such as
``strategy_controller`` or ``NeuralQuantile`` using standard PyTorch autograd.
"""

from typing import Iterable

import torch
from torch import nn


def soft_position(logits: torch.Tensor) -> torch.Tensor:
    """Convert action logits to a continuous position using softmax.

    Parameters
    ----------
    logits:
        Tensor of shape ``(T, 3)`` containing unnormalised logits for buy,
        sell and hold respectively.

    Returns
    -------
    torch.Tensor
        Continuous position in ``[-1, 1]`` for each timestep.
    """

    probs = torch.softmax(logits, dim=-1)
    # Buy prob minus sell prob gives a smooth position
    return probs[..., 0] - probs[..., 1]


def simulate_pnl(
    prices: torch.Tensor,
    position: torch.Tensor,
    slippage: float = 1e-4,
) -> torch.Tensor:
    """Simulate PnL for a continuous position vector.

    ``position`` should have the same length as ``prices``.  The function uses a
    very small slippage model where changing the position incurs a linear cost
    proportional to the absolute turnover.

    Parameters
    ----------
    prices: torch.Tensor
        Tensor of mid prices of length ``T``.
    position: torch.Tensor
        Continuous position for each timestep in ``[-1, 1]``.
    slippage: float, optional
        Cost coefficient applied to the absolute change in position.

    Returns
    -------
    torch.Tensor
        PnL for each step ``t -> t+1`` with shape ``(T-1,)``.
    """

    prices = prices.to(torch.float32)
    position = position.to(torch.float32)
    if prices.ndim != 1 or position.ndim != 1:
        raise ValueError("prices and position must be 1D tensors")
    if len(prices) != len(position):
        raise ValueError("prices and position must be the same length")

    # Price returns and position held during the interval
    returns = prices[1:] - prices[:-1]
    pos = position[:-1]

    # Turnover based cost – smooth proxy for slippage / spread
    turnover = torch.abs(position[1:] - position[:-1])
    cost = slippage * turnover

    pnl = pos * returns - cost
    return pnl


class DiffBacktestLoss(nn.Module):
    """PyTorch loss that backpropagates through the differentiable backtest."""

    def __init__(self, slippage: float = 1e-4):
        super().__init__()
        self.slippage = slippage

    def forward(
        self, prices: torch.Tensor, actions_or_pos: torch.Tensor
    ) -> torch.Tensor:
        """Return the *negative* average PnL for optimisation.

        ``actions_or_pos`` can either be a tensor of shape ``(T, 3)`` containing
        action logits (buy, sell, hold) or a direct position tensor of shape
        ``(T,)``.  In the former case a softmax is applied to obtain the
        position.
        """

        if actions_or_pos.ndim == 2:
            position = soft_position(actions_or_pos)
        else:
            position = actions_or_pos
        pnl = simulate_pnl(prices, position, self.slippage)
        # We minimise negative PnL to maximise profits
        return -pnl.mean()


__all__ = ["soft_position", "simulate_pnl", "DiffBacktestLoss"]
