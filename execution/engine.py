"""Simple execution engine that supports different scheduling strategies."""

from __future__ import annotations

import logging
import asyncio
from collections import deque
from time import perf_counter
from typing import TYPE_CHECKING, Callable, Deque, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from brokers import connection_manager as conn_mgr
from analysis.broker_tca import broker_tca

from .algorithms import (
    twap_schedule,
    vwap_schedule,
    twap_schedule_async,
    vwap_schedule_async,
    simple_slicer,
)
from .rl_executor import RLExecutor
from .execution_optimizer import ExecutionOptimizer, OptimizationLoopHandle
from .fill_history import record_fill
from metrics import SLIPPAGE_BPS, REALIZED_SLIPPAGE_BPS
from event_store.event_writer import record as record_event
from model_registry import ModelRegistry

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

if TYPE_CHECKING:  # pragma: no cover
    from models.capital_allocator import CapitalAllocator


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
        registry: Optional[ModelRegistry] = None,
        capital_allocator: Optional["CapitalAllocator"] = None,
    ) -> None:
        self.recent_volume: Deque[float] = deque(maxlen=volume_window)
        self.rl_executor = rl_executor
        self.rl_threshold = rl_threshold
        self.optimizer = optimizer or ExecutionOptimizer()
        self.capital_allocator = capital_allocator
        self.strategy_weights: Dict[str, float] = {}
        # Registry used to dynamically load RL policies on demand.  When
        # provided, an ``RLExecutor`` will be instantiated lazily when the
        # ``rl`` strategy is requested.
        self.registry = registry
        # Queue used to emit fill or cancellation events for asynchronous
        # execution.  Tests consume from this queue to verify event ordering.
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self._optimizer_handle: Optional[OptimizationLoopHandle] = None

    # ------------------------------------------------------------------
    def start_optimizer(self) -> Optional[OptimizationLoopHandle]:
        """Start the nightly optimiser loop if configured."""

        if self.optimizer is None:
            return None
        if self._optimizer_handle is not None:
            return self._optimizer_handle
        try:
            handle = self.optimizer.schedule_nightly()
        except Exception:
            handle = None
        self._optimizer_handle = handle
        return handle

    # ------------------------------------------------------------------
    def stop_optimizer(self) -> None:
        """Stop the nightly optimiser loop if it is running."""

        handle = self._optimizer_handle
        if not handle:
            return
        try:
            handle.stop()
            handle.join()
        except Exception:
            pass
        finally:
            self._optimizer_handle = None

    # ------------------------------------------------------------------
    def rebalance_capital(
        self, pnl: Dict[str, float], risk: Dict[str, float]
    ) -> Dict[str, float]:
        """Recompute capital weights using the allocator, if present."""
        if self.capital_allocator is None:
            return {}
        self.strategy_weights = self.capital_allocator.allocate(pnl, risk)
        return self.strategy_weights

    # ------------------------------------------------------------------
    def record_volume(self, volume: float) -> None:
        """Record recent traded volume for VWAP scheduling."""
        self.recent_volume.append(volume)

    # ------------------------------------------------------------------
    async def place_order(
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
        depth_cb: Optional[Callable[[], Tuple[float, float]]] = None,
        slippage_model: Optional[Callable[[float, str], float]] = None,
        limit_price: Optional[float] = None,
        limit_offset: Optional[float] = None,
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
            ``"ioc"`` (immediate-or-cancel), ``"twap"``, ``"vwap``" or ``"limit"``.
        expected_slippage_bps: float, optional
            Configured slippage assumption for logging/metrics.
        slippage_model: Callable, optional
            When provided, called with ``(quantity, side)`` to estimate slippage
            in basis points prior to execution.  Overrides ``expected_slippage_bps``.
        limit_price: float, optional
            Absolute price for limit orders.  If omitted, ``limit_offset`` is
            applied to the current best bid/ask depending on ``side``.
        limit_offset: float, optional
            Offset applied to bid/ask when ``limit_price`` is not supplied.
            Positive values make the order more aggressive for buys and less
            aggressive for sells.
        """

        order_ts = pd.Timestamp.utcnow()
        start = perf_counter()
        strat = strategy.lower()
        sign = 1 if side.lower() == "buy" else -1

        if slippage_model is not None:
            try:
                expected_slippage_bps = float(slippage_model(quantity, side))
            except Exception:
                # fall back to provided assumption when estimation fails
                pass

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
        if limit_price is not None:
            order_payload["limit_price"] = limit_price
        try:
            record_event("order", order_payload)
        except Exception:
            pass

        params = (
            self.optimizer.get_params()
            if self.optimizer
            else {"limit_offset": 0.0, "slice_size": None}
        )
        opt_limit_offset = float(params.get("limit_offset", 0.0) or 0.0)
        slice_size = params.get("slice_size")
        if limit_offset is None:
            limit_offset = opt_limit_offset

        use_rl = False
        if strat == "rl":
            if self.rl_executor is None and self.registry is not None:
                try:
                    path = self.registry.get_policy_path()
                except Exception:
                    path = None
                self.rl_executor = RLExecutor()
                if path:
                    try:
                        self.rl_executor.load(path)
                    except Exception:  # pragma: no cover - loading is best effort
                        self.rl_executor = None
            use_rl = self.rl_executor is not None
        elif self.rl_executor is not None:
            tier = getattr(monitor.capabilities, "capability_tier", lambda: "lite")()
            if (
                tier != "lite"
                and quantity >= self.rl_threshold
                and strat not in {"twap", "vwap"}
            ):
                use_rl = True

        if use_rl and self.rl_executor is not None:
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
            slices = await vwap_schedule_async(quantity, self.recent_volume)
        elif strat == "twap":
            intervals = max(len(self.recent_volume), 1)
            slices = await twap_schedule_async(quantity, intervals)
        elif strat == "limit":
            if slice_size:
                slices = simple_slicer(quantity, slice_size)
            else:
                intervals = max(len(self.recent_volume), 1)
                slices = await twap_schedule_async(quantity, intervals)
        elif slice_size:
            slices = simple_slicer(quantity, slice_size)
        else:
            slices = [quantity]

        # Closure returning latest market depth.  When ``depth_cb`` is not
        # supplied the initial bid/ask volumes are used.
        static_depth = (bid_vol, ask_vol)

        def _depth() -> Tuple[float, float]:
            return depth_cb() if depth_cb else static_depth

        filled = 0.0
        price_accum = 0.0
        for qty in slices:
            if strat == "limit":
                lp = (
                    limit_price
                    if limit_price is not None
                    else (ask if side.lower() == "buy" else bid)
                    + (limit_offset or 0.0) * (1 if side.lower() == "buy" else -1)
                )
                fill_qty, price_part = await self._execute_limit_slice(
                    qty, side, bid, ask, lp, _depth
                )
            else:
                fill_qty, price_part = await self._execute_slice(
                    qty,
                    side,
                    bid,
                    ask,
                    expected_slippage_bps,
                    limit_offset or 0.0,
                    _depth,
                )
            filled += fill_qty
            price_accum += price_part
            # Yield control to allow callers to modify depth between slices
            await asyncio.sleep(0)

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

    # ------------------------------------------------------------------
    async def _execute_slice(
        self,
        qty: float,
        side: str,
        bid: float,
        ask: float,
        expected_slippage_bps: float,
        limit_offset: float,
        depth_fn: Callable[[], Tuple[float, float]],
    ) -> Tuple[float, float]:
        """Send a child order to the broker and wait for acknowledgement.

        The coroutine determines the available depth via ``depth_fn`` which
        allows callers to dynamically update market conditions.  Events are
        placed onto :attr:`event_queue` to signal fills or cancellations.
        Returns the filled quantity and price contribution for averaging.
        """

        # Simulate an asynchronous broker round-trip
        await asyncio.sleep(0)
        bid_vol, ask_vol = depth_fn()
        if side.lower() == "buy":
            avail = ask_vol
            price = ask
            sign = 1
        else:
            avail = bid_vol
            price = bid
            sign = -1

        fill_qty = min(qty, avail)
        if fill_qty <= 0:
            await self.event_queue.put({"type": "cancel", "qty": 0})
            return 0.0, 0.0

        adj_price = price * (1 + sign * (expected_slippage_bps + limit_offset) / 10000.0)
        await self.event_queue.put(
            {
                "type": "fill",
                "qty": fill_qty,
                "price": adj_price,
                "partial": fill_qty < qty,
            }
        )
        return fill_qty, fill_qty * adj_price

    # ------------------------------------------------------------------
    async def _execute_limit_slice(
        self,
        qty: float,
        side: str,
        bid: float,
        ask: float,
        limit_price: float,
        depth_fn: Callable[[], Tuple[float, float]],
    ) -> Tuple[float, float]:
        """Simulate a limit child order and emit fill/cancel events."""

        await asyncio.sleep(0)
        bid_vol, ask_vol = depth_fn()
        if side.lower() == "buy":
            avail = ask_vol
            can_fill = limit_price >= ask
        else:
            avail = bid_vol
            can_fill = limit_price <= bid

        if not can_fill or avail <= 0:
            await self.event_queue.put({"type": "cancel", "qty": qty})
            return 0.0, 0.0

        fill_qty = min(qty, avail)
        await self.event_queue.put(
            {
                "type": "fill",
                "qty": fill_qty,
                "price": limit_price,
                "partial": fill_qty < qty,
            }
        )
        if fill_qty < qty:
            await self.event_queue.put({"type": "cancel", "qty": qty - fill_qty})
        return fill_qty, fill_qty * limit_price
