from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import asyncio
import os

import numpy as np
from scheduler import start_scheduler


@dataclass
class RiskMetrics:
    exposure: float = 0.0
    daily_loss: float = 0.0
    var: float = 0.0
    trading_halted: bool = False


class RiskManager:
    """Aggregate risk metrics from trade/PnL events."""

    def __init__(self, max_drawdown: float, max_var: float = float("inf"), var_window: int = 100) -> None:
        self.max_drawdown = max_drawdown
        self.max_var = max_var
        self.var_window = var_window
        self.metrics = RiskMetrics()
        self._pnl_history: List[float] = []

    def update(self, bot_id: str, pnl: float, exposure: float = 0.0) -> None:
        """Record a trade or PnL update from ``bot_id``."""
        self.metrics.exposure += exposure
        self.metrics.daily_loss += pnl
        self._pnl_history.append(pnl)
        if len(self._pnl_history) > self.var_window:
            self._pnl_history.pop(0)
        if self._pnl_history:
            self.metrics.var = float(-np.percentile(self._pnl_history, 1))
        if self.metrics.daily_loss <= -self.max_drawdown or self.metrics.var > self.max_var:
            self.metrics.trading_halted = True

    async def monitor(self, queue: "asyncio.Queue[tuple[str, float, float]]") -> None:
        """Consume trade/PnL events from ``queue`` and update metrics."""
        while True:
            bot_id, pnl, exposure = await queue.get()
            self.update(bot_id, pnl, exposure)

    def status(self) -> Dict[str, float | bool]:
        """Return current aggregated risk metrics."""
        return {
            "exposure": self.metrics.exposure,
            "daily_loss": self.metrics.daily_loss,
            "var": self.metrics.var,
            "trading_halted": self.metrics.trading_halted,
        }

    def halt(self) -> None:
        self.metrics.trading_halted = True

    def reset(self) -> None:
        self.metrics = RiskMetrics()
        self._pnl_history.clear()


MAX_DRAWDOWN = float(os.getenv("MAX_PORTFOLIO_DRAWDOWN", "1e9"))
MAX_VAR = float(os.getenv("MAX_VAR", "1e9"))

risk_manager = RiskManager(MAX_DRAWDOWN, MAX_VAR)
start_scheduler()
