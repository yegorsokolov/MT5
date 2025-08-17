from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, TYPE_CHECKING
import asyncio
import os

import numpy as np
import pandas as pd
from scheduler import start_scheduler
from risk import risk_of_ruin
from risk.budget_allocator import BudgetAllocator
from analytics.metrics_store import record_metric
try:
    from utils.alerting import send_alert
except Exception:  # pragma: no cover - utils may be stubbed in tests
    def send_alert(msg: str) -> None:  # type: ignore
        return

if TYPE_CHECKING:  # pragma: no cover - used only for typing
    from risk.tail_hedger import TailHedger


@dataclass
class RiskMetrics:
    exposure: float = 0.0
    daily_loss: float = 0.0
    var: float = 0.0
    risk_of_ruin: float = 0.0
    trading_halted: bool = False
    factor_contributions: Dict[str, float] = field(default_factory=dict)


class RiskManager:
    """Aggregate risk metrics from trade/PnL events."""

    def __init__(
        self,
        max_drawdown: float,
        max_var: float = float("inf"),
        var_window: int = 100,
        tail_hedger: "TailHedger" | None = None,
        risk_of_ruin_threshold: float = 1.0,
        initial_capital: float = 1.0,
    ) -> None:
        self.max_drawdown = max_drawdown
        self.max_var = max_var
        self.var_window = var_window
        self.metrics = RiskMetrics()
        self._pnl_history: List[float] = []
        self._factor_history: List[Dict[str, float]] = []
        self._bot_pnl_history: Dict[str, List[float]] = {}
        self.budget_allocator = BudgetAllocator(initial_capital)
        self.tail_hedger = tail_hedger
        self.risk_of_ruin_threshold = risk_of_ruin_threshold
        self.initial_capital = initial_capital

    def attach_tail_hedger(self, hedger: "TailHedger") -> None:
        """Attach a :class:`~risk.tail_hedger.TailHedger` instance."""
        self.tail_hedger = hedger

    def update(
        self,
        bot_id: str,
        pnl: float,
        exposure: float = 0.0,
        check_hedge: bool = True,
        factor_returns: Dict[str, float] | None = None,
    ) -> None:
        """Record a trade or PnL update from ``bot_id``."""
        self.metrics.exposure += exposure
        self.metrics.daily_loss += pnl
        self._pnl_history.append(pnl)
        bot_hist = self._bot_pnl_history.setdefault(bot_id, [])
        bot_hist.append(pnl)
        if factor_returns is not None:
            self._factor_history.append(factor_returns)
        if len(self._pnl_history) > self.var_window:
            self._pnl_history.pop(0)
            if self._factor_history:
                self._factor_history.pop(0)
        if len(bot_hist) > self.var_window:
            bot_hist.pop(0)
        if self._pnl_history:
            self.metrics.var = float(-np.percentile(self._pnl_history, 1))
            returns = pd.Series(self._pnl_history) / self.initial_capital
            self.metrics.risk_of_ruin = float(
                risk_of_ruin(returns, self.initial_capital)
            )
            if self._factor_history and len(self._factor_history) == len(self._pnl_history):
                factors_df = pd.DataFrame(self._factor_history)
                from portfolio.factor_risk import FactorRisk

                fr = FactorRisk(factors_df)
                contrib = fr.factor_contributions(returns)
                self.metrics.factor_contributions = contrib.to_dict()
        breach_reason = None
        if self.metrics.daily_loss <= -self.max_drawdown:
            breach_reason = (
                f"max drawdown exceeded: {self.metrics.daily_loss:.2f}"
                f" <= {-self.max_drawdown}"
            )
        elif self.metrics.var > self.max_var:
            breach_reason = (
                f"VaR limit exceeded: {self.metrics.var:.2f} > {self.max_var}"
            )
        elif self.metrics.risk_of_ruin > self.risk_of_ruin_threshold:
            breach_reason = (
                "risk of ruin exceeded: "
                f"{self.metrics.risk_of_ruin:.2f} > {self.risk_of_ruin_threshold}"
            )
        if breach_reason:
            self.metrics.trading_halted = True
            send_alert(f"Risk limit breached: {breach_reason}")
        if check_hedge and self.tail_hedger is not None:
            self.tail_hedger.evaluate()
        # Record metrics for analytics
        try:
            record_metric("pnl", pnl, {"bot_id": bot_id})
            record_metric("exposure", self.metrics.exposure)
            record_metric("var", self.metrics.var)
            record_metric("risk_of_ruin", self.metrics.risk_of_ruin)
        except Exception:
            pass

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
            "risk_of_ruin": self.metrics.risk_of_ruin,
            "trading_halted": self.metrics.trading_halted,
            "factor_contributions": self.metrics.factor_contributions,
        }

    def halt(self) -> None:
        self.metrics.trading_halted = True

    def reset(self) -> None:
        self.metrics = RiskMetrics()
        self._pnl_history.clear()
        self._factor_history.clear()
        self._bot_pnl_history.clear()
        self.budget_allocator.budgets.clear()

    # Risk budgets ----------------------------------------------------
    def rebalance_budgets(self) -> Dict[str, float]:
        """Recompute risk budgets from per-bot PnL history."""
        returns = {
            bot: pd.Series(hist) / self.initial_capital
            for bot, hist in self._bot_pnl_history.items()
            if hist
        }
        return self.budget_allocator.allocate(returns)

    def adjust_position_size(self, bot_id: str, size: float) -> float:
        """Scale ``size`` by the risk budget for ``bot_id``."""
        frac = self.budget_allocator.fraction(bot_id)
        return size * frac


from risk.tail_hedger import TailHedger

MAX_DRAWDOWN = float(os.getenv("MAX_PORTFOLIO_DRAWDOWN", "1e9"))
MAX_VAR = float(os.getenv("MAX_VAR", "1e9"))
TAIL_HEDGE_VAR = float(os.getenv("TAIL_HEDGE_VAR", str(MAX_VAR)))
RISK_OF_RUIN_THRESHOLD = float(os.getenv("RISK_OF_RUIN_THRESHOLD", "1.0"))
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "1.0"))

risk_manager = RiskManager(
    MAX_DRAWDOWN,
    MAX_VAR,
    risk_of_ruin_threshold=RISK_OF_RUIN_THRESHOLD,
    initial_capital=INITIAL_CAPITAL,
)
tail_hedger = TailHedger(risk_manager, TAIL_HEDGE_VAR)
risk_manager.attach_tail_hedger(tail_hedger)
start_scheduler()
