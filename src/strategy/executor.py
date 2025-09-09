from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict

from src.modes import Mode


@dataclass
class StrategyExecutor:
    """Simple executor enforcing mode restrictions for live trading."""

    mode: Mode
    strategy: Dict[str, Any] = field(default_factory=dict)
    metadata_path: Path = Path(__file__).resolve().parents[2] / "strategies" / "metadata.json"

    def _approved(self) -> bool:
        """Return ``True`` if the strategy is approved for live trading."""

        name = self.strategy.get("name")
        if not name:
            return bool(self.strategy.get("approved"))

        try:
            data = json.loads(self.metadata_path.read_text())
        except FileNotFoundError:
            return False

        return bool(data.get(name, {}).get("approved"))

    def place_live_order(self, order: Any) -> str:
        """Attempt to place a live order.

        Raises:
            PermissionError: If live trading is not permitted in the current mode
                or the strategy is not approved in ``metadata.json``.
        """

        if self.mode is not Mode.LIVE_TRADING or not self._approved():
            raise PermissionError("Live trading not permitted")

        # In a real system, this would send the order to a broker.
        return "order_placed"

    def on_tick(self, tick: Dict[str, Any]) -> None:
        """Process a market tick.

        An order is generated on every tick to allow continual training. Orders are
        only sent to the broker in live mode with an approved strategy; otherwise the
        resulting profit or loss is calculated for training without execution.
        """

        generate_order = self.strategy.get("generate_order")
        update = self.strategy.get("update")
        if not callable(generate_order) or not callable(update):
            return

        order = generate_order(tick)
        price = tick.get("price", 0)
        next_price = tick.get("next_price", price)
        quantity = order.get("quantity", 1) if isinstance(order, dict) else 1
        outcome = (next_price - price) * quantity
        update(order, outcome)

        if self.mode is Mode.LIVE_TRADING and self._approved():
            self.place_live_order(order)
