"""Simple execution engine that supports different scheduling strategies."""

from __future__ import annotations

import logging
from collections import deque
from time import perf_counter
from typing import Deque, Iterable, List, Optional

import pandas as pd

from brokers import connection_manager as conn_mgr
from analysis.broker_tca import broker_tca

from .algorithms import twap_schedule, vwap_schedule
from .rl_executor import RLExecutor
from .execution_optimizer import ExecutionOptimizer
from .fill_history import record_fill
from metrics import SLIPPAGE_BPS, REALIZED_SLIPPAGE_BPS
from event_store.event_writer import record as record_event

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
        optimizer: Optional[ExecutionOptimizer] = None,
    ) -> None:
        self.recent_volume: Deque[float] = deque(maxlen=volume_window)
        self.rl_executor = rl_executor
        self.rl_threshold = rl_threshold
        self.optimizer = optimizer or ExecutionOptimizer()
        try:
            self.optimizer.schedule_nightly()
        except Exception:
            pass

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

        order_ts = pd.Timestamp.utcnow()
        start = perf_counter()
        strat = strategy.lower()

        order_payload = {
            "side": side,
            "quantity": quantity,
            "bid": bid,
            "ask": ask,
            "bid_vol": bid_vol,
            "ask_vol": ask_vol,
            "mid": mid,
            "strategy": strat,
            "expected_slippage_bps": expected_slippage_bps,
        }
        try:
            record_event("order", order_payload)
        except Exception:
            pass

        params = (
            self.optimizer.get_params()
            if self.optimizer
            else {"limit_offset": 0.0, "slice_size": None}
        )
        limit_offset = float(params.get("limit_offset", 0.0) or 0.0)
        slice_size = params.get("slice_size")

        use_rl = False
        if self.rl_executor is not None:
            tier = getattr(monitor.capabilities, "capability_tier", lambda: "lite")()
            if strat == "rl" or (
                tier != "lite"
                and quantity >= self.rl_threshold
                and strat not in {"twap", "vwap"}
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
        elif slice_size:
            slices: List[float] = []
            remaining = quantity
            while remaining > 0:
                take = min(remaining, slice_size)
                slices.append(take)
                remaining -= take
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
            adj_price = price * (
                1 + sign * (expected_slippage_bps + limit_offset) / 10000.0
            )
            fill_qty = min(qty, avail)
            if fill_qty <= 0:
                continue
            filled += fill_qty
            price_accum += fill_qty * adj_price

        avg_price = (
            price_accum / filled if filled else (ask if side.lower() == "buy" else bid)
        )
        realized = (avg_price - mid) / mid * sign * 10000.0 if mid else 0.0
        latency = perf_counter() - start
        depth = ask_vol if side.lower() == "buy" else bid_vol
        fill_ts = order_ts + pd.to_timedelta(latency, unit="s")
        try:
            broker = conn_mgr.get_active_broker()
            name = getattr(broker, "__name__", broker.__class__.__name__)
            broker_tca.record(name, order_ts, fill_ts, realized)
        except Exception:
            pass
        try:
            record_fill(slippage=realized, latency=latency, depth=depth)
        except Exception:
            pass
        try:
            record_event(
                "fill",
                {
                    **order_payload,
                    "filled": filled,
                    "avg_price": avg_price,
                    "realized_slippage_bps": realized,
                },
            )
        except Exception:
            pass
        try:
            SLIPPAGE_BPS.set(expected_slippage_bps)
            REALIZED_SLIPPAGE_BPS.set(realized)
        except Exception:
            pass
        return {"avg_price": avg_price, "filled": filled}
