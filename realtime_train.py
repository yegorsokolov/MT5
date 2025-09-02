import asyncio
import time
from pathlib import Path
import os
import random
import logging
from typing import Callable, List

import numpy as np
import pandas as pd

from brokers import connection_manager as conn_mgr
from brokers import mt5_direct

from utils import load_config
from execution import ExecutionEngine, place_order
from utils.resource_monitor import monitor
try:
    from utils.alerting import send_alert
except Exception:  # pragma: no cover - utils may be stubbed in tests
    def send_alert(msg: str) -> None:  # type: ignore
        return

from data.features import make_features
from log_utils import setup_logging, log_exceptions
from signal_queue import get_signal_backend
from data.sanitize import sanitize_ticks
from data.feature_scaler import FeatureScaler
from metrics import (
    RECONNECT_COUNT,
    FEATURE_ANOMALIES,
    RESOURCE_RESTARTS,
    QUEUE_DEPTH,
    BATCH_LATENCY,
)

from telemetry import get_tracer, get_meter
from analytics.metrics_store import record_metric, TS_PATH
from analysis.data_quality import apply_quality_checks
from analysis.domain_adapter import DomainAdapter

tracer = get_tracer(__name__)
meter = get_meter(__name__)
_ticks_counter = meter.create_counter(
    "ticks_fetched", description="Total ticks fetched"  # type: ignore[arg-type]
)
_batch_latency = meter.create_histogram(
    "batch_latency_seconds",
    unit="s",
    description="Latency for processing tick batches",
)
_empty_batch_count = 0

setup_logging()
logger = logging.getLogger(__name__)

# initialize connection manager with primary MetaTrader5 broker
conn_mgr.init([mt5_direct])
exec_engine = ExecutionEngine()

try:  # pragma: no cover - orchestrator may be stubbed in tests
    from core.orchestrator import Orchestrator
    orchestrator = Orchestrator.start()
    watchdog = orchestrator.monitor
except Exception:  # pragma: no cover
    import types

    orchestrator = None
    watchdog = types.SimpleNamespace(max_rss_mb=None, max_cpu_pct=None, capabilities=None)

if os.name != "nt" and getattr(watchdog, "capabilities", None):
    if watchdog.capabilities.cpus > 1:  # pragma: no cover - depends on host
        try:
            import uvloop  # type: ignore

            uvloop.install()
            logger.info("uvloop installed")
        except Exception:  # pragma: no cover - uvloop optional
            logger.info("uvloop not available; using default event loop")


async def _handle_resource_breach(reason: str) -> None:
    logger.error("Resource watchdog triggered: %s", reason)
    RESOURCE_RESTARTS.inc()
    send_alert(f"Resource watchdog triggered: {reason}")
    os._exit(1)


async def fetch_ticks(symbol: str, n: int = 1000, retries: int = 3) -> pd.DataFrame:
    """Fetch recent tick data from the active broker asynchronously."""
    cfg = load_config()
    max_empty = cfg.get("max_empty_batches", 3)
    global _empty_batch_count
    with tracer.start_as_current_span("fetch_ticks"):
        for attempt in range(retries):
            broker = conn_mgr.get_active_broker()
            ticks = await asyncio.to_thread(
                broker.copy_ticks_from, symbol, int(time.time()) - n, n, broker.COPY_TICKS_ALL
            )
            if ticks is None:
                logger.warning(
                    "Failed to fetch ticks for %s on attempt %d, attempting failover", symbol, attempt + 1
                )
                RECONNECT_COUNT.inc()
                conn_mgr.failover()
                await asyncio.sleep(1)
                continue
            if len(ticks) == 0:
                _empty_batch_count += 1
                record_metric("fetch_empty_batches", _empty_batch_count, tags={"symbol": symbol})
                if _empty_batch_count >= max_empty:
                    logger.error(
                        "No ticks received for %s after %d empty batches", symbol, _empty_batch_count
                    )
                    send_alert(
                        f"No ticks received for {symbol} after {_empty_batch_count} attempts"
                    )
                    RECONNECT_COUNT.inc()
                    conn_mgr.failover()
                    _empty_batch_count = 0
                    await asyncio.sleep(1)
                    continue
                return pd.DataFrame()
            _empty_batch_count = 0
            df = pd.DataFrame(ticks)
            df["Timestamp"] = pd.to_datetime(df["time"], unit="s")
            df.rename(columns={"bid": "Bid", "ask": "Ask", "volume": "Volume"}, inplace=True)
            df = df[["Timestamp", "Bid", "Ask", "Volume"]]
            df["BidVolume"] = df["Volume"]
            df["AskVolume"] = df["Volume"]
            df.drop(columns=["Volume"], inplace=True)
            df = sanitize_ticks(df)
            df, report = apply_quality_checks(df)
            if any(report.values()):
                logger.info("Realtime data quality: %s", report)
                for k, v in report.items():
                    record_metric(f"data_quality_{k}", v, tags={"source": "realtime"})
            _ticks_counter.add(len(df))
            return df
        return pd.DataFrame()


async def generate_features(context: pd.DataFrame) -> pd.DataFrame:
    """Asynchronously generate features from context data."""
    with tracer.start_as_current_span("generate_features"):
        if context.empty:
            return pd.DataFrame()
        df = await asyncio.to_thread(make_features, context)
        return df


def apply_liquidity_adjustment(
    df: pd.DataFrame, metrics_path: Path | None = None
) -> pd.DataFrame:
    """Append simulated fill prices based on liquidity metrics.

    The function also records aggregate slippage and liquidity usage metrics
    using :func:`record_metric`.
    """
    if metrics_path is None:
        metrics_path = TS_PATH
    if {"mid", "vw_spread", "market_impact"}.issubset(df.columns):
        buy = df["mid"] + df["vw_spread"] / 2 + df["market_impact"]
        sell = df["mid"] - df["vw_spread"] / 2 - df["market_impact"]
        df = df.copy()
        df["buy_fill"] = buy
        df["sell_fill"] = sell
        record_metric(
            "slippage",
            float((buy - df["mid"]).abs().mean()),
            path=metrics_path,
        )
        record_metric(
            "liquidity_usage",
            float(df.get("depth_imbalance", pd.Series(0, index=df.index)).abs().mean()),
            path=metrics_path,
        )
    return df


async def dispatch_signals(queue, df: pd.DataFrame) -> None:
    """Asynchronously dispatch signals to the provided queue."""
    with tracer.start_as_current_span("dispatch_signals"):
        if queue is None or df.empty:
            return
        await asyncio.to_thread(queue.publish_dataframe, df)

        tier = getattr(monitor, "capability_tier", "lite")
        strategy = "ioc" if tier == "lite" else "vwap"
        for row in df.itertuples(index=False):
            side = getattr(row, "side", None)
            size = getattr(row, "size", None)
            if side is None or size is None:
                continue
            bid = getattr(row, "Bid", getattr(row, "mid", 0.0))
            ask = getattr(row, "Ask", getattr(row, "mid", 0.0))
            bid_vol = getattr(row, "BidVolume", float("inf"))
            ask_vol = getattr(row, "AskVolume", float("inf"))
            mid = getattr(row, "mid", (bid + ask) / 2 if (bid and ask) else bid)
            exec_engine.record_volume(bid_vol + ask_vol)
            exec_engine.place_order(
                side=side,
                quantity=size,
                bid=bid,
                ask=ask,
                bid_vol=bid_vol,
                ask_vol=ask_vol,
                mid=mid,
                strategy=strategy,
            )
            symbol = getattr(row, "Symbol", getattr(row, "symbol", None))
            if symbol:
                place_order(symbol=symbol, side=side, volume=size)


async def tick_producer(
    symbols: List[str],
    queue: asyncio.Queue,
    fetch_fn: Callable = fetch_ticks,
    throttle_threshold: int = 1000,
) -> None:
    """Continuously fetch ticks for symbols and enqueue them."""
    while True:
        with tracer.start_as_current_span("tick_producer_loop"):
            if queue.qsize() > throttle_threshold:
                await asyncio.sleep(0.1)
                continue
            tick_frames = []
            fetch_results = await asyncio.gather(*(fetch_fn(sym, 500) for sym in symbols))
            for sym, ticks in zip(symbols, fetch_results):
                if not ticks.empty:
                    ticks["Symbol"] = sym
                    tick_frames.append(ticks)
            if tick_frames:
                batch = pd.concat(tick_frames, ignore_index=True)
                batch["Timestamp"] = pd.to_datetime(batch["Timestamp"])
                await queue.put(batch)
                QUEUE_DEPTH.set(queue.qsize())
            else:
                await asyncio.sleep(0.1)


async def tick_worker(
    queue: asyncio.Queue,
    process_batch: Callable[[pd.DataFrame], asyncio.Future],
    *,
    target_latency: float = 1.0,
    min_batch: int = 1,
    max_batch: int = 32,
) -> None:
    """Process queued ticks in batches adapting batch size to latency."""
    batch_size = min_batch
    batch: List[pd.DataFrame] = []
    while True:
        frame = await queue.get()
        queue.task_done()
        batch.append(frame)
        while len(batch) < batch_size:
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=0.01)
                queue.task_done()
                batch.append(frame)
            except asyncio.TimeoutError:
                break
        start = time.perf_counter()
        with tracer.start_as_current_span("process_batch"):
            await process_batch(pd.concat(batch, ignore_index=True))
        latency = time.perf_counter() - start
        BATCH_LATENCY.set(latency)
        _batch_latency.record(latency)
        if latency > target_latency and batch_size > min_batch:
            batch_size = max(min_batch, batch_size // 2)
        elif latency < target_latency / 2 and batch_size < max_batch:
            batch_size = min(max_batch, batch_size * 2)
        batch.clear()
        QUEUE_DEPTH.set(queue.qsize())


@log_exceptions
async def train_realtime():
    cfg = load_config()
    with tracer.start_as_current_span("train_realtime"):
        if watchdog.max_rss_mb or watchdog.max_cpu_pct:
            watchdog.alert_callback = lambda msg: asyncio.create_task(
                _handle_resource_breach(msg)
            )
        seed = cfg.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        root = Path(__file__).resolve().parent
        scaler_path = root / "scaler.pkl"
        scaler = FeatureScaler.load(scaler_path)
        adapter_path = root / "domain_adapter.pkl"
        adapter = DomainAdapter.load(adapter_path)

        symbols = cfg.get("symbols") or [cfg.get("symbol", "EURUSD")]

        signal_backend = cfg.get("signal_backend", "none")
        queue = None
        if signal_backend in {"kafka", "redis"}:
            queue = get_signal_backend(cfg)
            logger.info("Using %s backend for signal queue", signal_backend)

        async def process_batch(batch: pd.DataFrame) -> None:
            feats = await generate_features(batch)
            if feats.empty:
                return
            feats = apply_liquidity_adjustment(feats)
            num_cols = feats.select_dtypes(np.number).columns
            if len(num_cols) > 0:
                scaler.partial_fit(feats[num_cols])
                feats[num_cols] = scaler.transform(feats[num_cols])
                feats[num_cols] = adapter.transform(feats[num_cols])
                scaler.save(scaler_path)
            await dispatch_signals(queue, feats)

        tick_queue: asyncio.Queue = asyncio.Queue()
        producer = asyncio.create_task(
            tick_producer(
                symbols,
                tick_queue,
                throttle_threshold=cfg.get("backlog_threshold", 1000),
            )
        )
        workers = [
            asyncio.create_task(
                tick_worker(
                    tick_queue,
                    process_batch,
                    target_latency=cfg.get("target_batch_latency", 1.0),
                )
            )
            for _ in range(cfg.get("worker_count", 1))
        ]
        await asyncio.gather(producer, *workers)


if __name__ == "__main__":
    asyncio.run(train_realtime())
