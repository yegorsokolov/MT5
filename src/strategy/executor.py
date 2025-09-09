from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from src.modes import Mode


@dataclass
class StrategyExecutor:
    """Simple executor enforcing mode restrictions for live trading."""

    mode: Mode
    strategy: Dict[str, Any] = field(default_factory=dict)

    def place_live_order(self, order: Any) -> str:
        """Attempt to place a live order.

        Raises:
            PermissionError: If live trading is not permitted in the current mode
                or the strategy is not approved.
        """

        if self.mode is not Mode.LIVE_TRADING or not self.strategy.get("approved"):
            raise PermissionError("Live trading not permitted")

        # In a real system, this would send the order to a broker.
        return "order_placed"
