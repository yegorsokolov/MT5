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
import MetaTrader5 as mt5  # type: ignore

from utils import load_config
try:
    from utils.alerting import send_alert
except Exception:  # pragma: no cover - utils may be stubbed in tests
    def send_alert(msg: str) -> None:  # type: ignore
        return
try:  # allow tests to stub out utils
    from utils.resource_monitor import ResourceMonitor
except Exception:  # pragma: no cover - fallback when utils is stubbed
    class ResourceMonitor:  # type: ignore
        def __init__(self, *a, **k):
            self.max_rss_mb = None
            self.max_cpu_pct = None

        def start(self) -> None:
            return

from data.features import make_features
from log_utils import setup_logging, log_exceptions
from signal_queue import get_signal_backend
from data.sanitize import sanitize_ticks
from metrics import (
    RECONNECT_COUNT,
    FEATURE_ANOMALIES,
    RESOURCE_RESTARTS,
    QUEUE_DEPTH,
    BATCH_LATENCY,
)

setup_logging()
logger = logging.getLogger(__name__)

# initialize connection manager with primary MetaTrader5 broker
conn_mgr.init([mt5])

MAX_RSS_MB = float(os.getenv("MAX_RSS_MB", "0") or 0)
MAX_CPU_PCT = float(os.getenv("MAX_CPU_PCT", "0") or 0)
watchdog = ResourceMonitor(
    max_rss_mb=MAX_RSS_MB or None, max_cpu_pct=MAX_CPU_PCT or None
)


async def _handle_resource_breach(reason: str) -> None:
    logger.error("Resource watchdog triggered: %s", reason)
    RESOURCE_RESTARTS.inc()
    send_alert(f"Resource watchdog triggered: {reason}")
    os._exit(1)


async def fetch_ticks(symbol: str, n: int = 1000, retries: int = 3) -> pd.DataFrame:
    """Fetch recent tick data from the active broker asynchronously."""
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
            return pd.DataFrame()
        df = pd.DataFrame(ticks)
        df["Timestamp"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"bid": "Bid", "ask": "Ask", "volume": "Volume"}, inplace=True)
        df = df[["Timestamp", "Bid", "Ask", "Volume"]]
        df["BidVolume"] = df["Volume"]
        df["AskVolume"] = df["Volume"]
        df.drop(columns=["Volume"], inplace=True)
        df = sanitize_ticks(df)
        return df
    return pd.DataFrame()


async def generate_features(context: pd.DataFrame) -> pd.DataFrame:
    """Asynchronously generate features from context data."""
    if context.empty:
        return pd.DataFrame()
    return await asyncio.to_thread(make_features, context)


async def dispatch_signals(queue, df: pd.DataFrame) -> None:
    """Asynchronously dispatch signals to the provided queue."""
    if queue is None or df.empty:
        return
    await asyncio.to_thread(queue.publish_dataframe, df)


async def tick_producer(
    symbols: List[str],
    queue: asyncio.Queue,
    fetch_fn: Callable = fetch_ticks,
    throttle_threshold: int = 1000,
) -> None:
    """Continuously fetch ticks for symbols and enqueue them."""
    while True:
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
        await process_batch(pd.concat(batch, ignore_index=True))
        latency = time.perf_counter() - start
        BATCH_LATENCY.set(latency)
        if latency > target_latency and batch_size > min_batch:
            batch_size = max(min_batch, batch_size // 2)
        elif latency < target_latency / 2 and batch_size < max_batch:
            batch_size = min(max_batch, batch_size * 2)
        batch.clear()
        QUEUE_DEPTH.set(queue.qsize())


@log_exceptions
async def train_realtime():
    cfg = load_config()
    if watchdog.max_rss_mb or watchdog.max_cpu_pct:
        watchdog.alert_callback = lambda msg: asyncio.create_task(
            _handle_resource_breach(msg)
        )
        watchdog.start()
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    symbols = cfg.get("symbols") or [cfg.get("symbol", "EURUSD")]

    signal_backend = cfg.get("signal_backend", "zmq")
    queue = None
    if signal_backend in {"kafka", "redis"}:
        queue = get_signal_backend(cfg)
        logger.info("Using %s backend for signal queue", signal_backend)

    async def process_batch(batch: pd.DataFrame) -> None:
        feats = await generate_features(batch)
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
