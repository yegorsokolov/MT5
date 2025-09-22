"""Tail risk hedging utilities.

This module provides a simple ``TailHedger`` that monitors portfolio
Value-at-Risk (VaR) via a :class:`~risk_manager.RiskManager` instance and
adds protective hedges when risk exceeds a configurable threshold.

The hedger is intentionally minimal – it does not attempt to price
real-world options but instead records synthetic hedge trades that offset
portfolio exposure.  These trades are fed back into the risk manager so
hedges are evaluated and executed alongside normal trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, TYPE_CHECKING
import logging

if TYPE_CHECKING:  # pragma: no cover - used only for type checking
from mt5.risk_manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class TailHedger:
    """Monitor VaR and place protective hedges when thresholds are breached."""

    risk_manager: "RiskManager"
    var_threshold: float
    hedge_ratio: float = 1.0
    hedges: List[Dict[str, float]] = field(default_factory=list)

    def evaluate(self) -> bool:
        """Evaluate current risk and execute a hedge if required.

        Returns ``True`` if a hedge was executed.
        """

        var = self.risk_manager.metrics.var
        exposure = self.risk_manager.metrics.exposure
        logger.debug("Evaluating tail hedge: var=%.4f exposure=%.4f", var, exposure)
        if var <= self.var_threshold or exposure == 0:
            return False

        size = -exposure * self.hedge_ratio
        logger.info(
            "Tail hedge triggered: var=%.4f threshold=%.4f size=%.4f",
            var,
            self.var_threshold,
            size,
        )
        self.hedges.append({"size": size, "var": var})
        # Register hedge trade without triggering another hedge evaluation
        self.risk_manager.update("tail_hedge", 0.0, size, check_hedge=False)
        return True
