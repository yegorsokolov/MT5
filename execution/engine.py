"""Simple execution engine that supports different scheduling strategies."""
from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Iterable, List, Optional

from .algorithms import twap_schedule, vwap_schedule
from .rl_executor import RLExecutor
from metrics import SLIPPAGE_BPS, REALIZED_SLIPPAGE_BPS

try:  # optional dependency
    from utils.resource_monitor import monitor
except Exception:  # pragma: no cover - light fallback
    class _Cap:
        @staticmethod
        def capability_tier() -> str:
            return "lite"

    class _Mon:
        capabilities = _Cap()

    monitor = _Mon()  # type: ignore

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Route orders using basic execution algorithms.

    The engine keeps a rolling window of recent volumes which is used by the
    VWAP scheduler to apportion child order sizes.
    """

    def __init__(
        self,
        volume_window: int = 20,
        rl_executor: Optional[RLExecutor] = None,
        rl_threshold: float = 0.0,
    ) -> None:
        self.recent_volume: Deque[float] = deque(maxlen=volume_window)
        self.rl_executor = rl_executor
        self.rl_threshold = rl_threshold

    # ------------------------------------------------------------------
    def record_volume(self, volume: float) -> None:
        """Record recent traded volume for VWAP scheduling."""
        self.recent_volume.append(volume)

    # ------------------------------------------------------------------
    def place_order(
        self,
        *,
        side: str,
        quantity: float,
        bid: float,
        ask: float,
        bid_vol: float,
        ask_vol: float,
        mid: float,
        strategy: str = "ioc",
        expected_slippage_bps: float = 0.0,
    ) -> dict:
        """Execute an order using the requested ``strategy``.

        Parameters
        ----------
        side: str
            ``"buy"`` or ``"sell"``.
        quantity: float
            Parent order quantity.
        bid/ask: float
            Current best bid/ask.
        bid_vol/ask_vol: float
            Available volume at the best bid/ask.
        mid: float
            Mid price used for slippage calculations.
        strategy: str, optional
            ``"ioc"`` (immediate-or-cancel), ``"twap"`` or ``"vwap"``.
        expected_slippage_bps: float, optional
            Configured slippage assumption for logging/metrics.
        """

        strat = strategy.lower()

        use_rl = False
        if self.rl_executor is not None:
            tier = getattr(monitor.capabilities, "capability_tier", lambda: "lite")()
            if strat == "rl" or (
                tier != "lite" and quantity >= self.rl_threshold and strat not in {"twap", "vwap"}
            ):
                use_rl = True

        if use_rl:
            return self.rl_executor.execute(
                side=side,
                quantity=quantity,
                bid=bid,
                ask=ask,
                bid_vol=bid_vol,
                ask_vol=ask_vol,
                mid=mid,
            )

        if strat == "vwap" and self.recent_volume:
            slices = vwap_schedule(quantity, self.recent_volume)
        elif strat == "twap":
            intervals = max(len(self.recent_volume), 1)
            slices = twap_schedule(quantity, intervals)
        else:
            slices = [quantity]

        filled = 0.0
        price_accum = 0.0
        for qty in slices:
            if side.lower() == "buy":
                avail = ask_vol
                price = ask
                sign = 1
            else:
                avail = bid_vol
                price = bid
                sign = -1
            adj_price = price * (1 + sign * expected_slippage_bps / 10000.0)
            fill_qty = min(qty, avail)
            if fill_qty <= 0:
                continue
            filled += fill_qty
            price_accum += fill_qty * adj_price

        avg_price = price_accum / filled if filled else (ask if side.lower() == "buy" else bid)
        realized = (avg_price - mid) / mid * sign * 10000.0 if mid else 0.0
        logger.info(
            "Slippage expected %.2f bps vs realized %.2f bps", expected_slippage_bps, realized
        )
        try:
            SLIPPAGE_BPS.set(expected_slippage_bps)
            REALIZED_SLIPPAGE_BPS.set(realized)
        except Exception:
            pass
        return {"avg_price": avg_price, "filled": filled}
