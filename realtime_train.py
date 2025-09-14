import asyncio
import time
from pathlib import Path
import os
import random
import logging
import argparse
import sys
from typing import Callable, List

import numpy as np
import pandas as pd

from brokers import connection_manager as conn_mgr
from brokers import mt5_direct

from utils import load_config
from execution import ExecutionEngine, place_order
from utils.resource_monitor import monitor
from utils.graceful_exit import graceful_exit

try:
    from utils.alerting import send_alert
except Exception:  # pragma: no cover - utils may be stubbed in tests

    def send_alert(msg: str) -> None:  # type: ignore
        return


from data.features import make_features
from log_utils import setup_logging, log_exceptions
from signal_queue import get_signal_backend, publish_dataframe_async
from data.sanitize import sanitize_ticks
from data.feature_scaler import FeatureScaler
from event_store.event_writer import record as record_event
from metrics import (
    RECONNECT_COUNT,
    FEATURE_ANOMALIES,
    RESOURCE_RESTARTS,
    QUEUE_DEPTH,
    BATCH_LATENCY,
)

try:
    from telemetry import get_tracer, get_meter
except Exception:  # pragma: no cover - fallback if telemetry not installed

    def get_tracer(name: str):  # type: ignore
        class _Span:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        class _Tracer:
            def start_as_current_span(self, *a, **k):
                return _Span()

        return _Tracer()

    def get_meter(name: str):  # type: ignore
        class _Counter:
            def add(self, *a, **k):
                return None

        class _Histogram:
            def record(self, *a, **k):
                return None

        class _Meter:
            def create_counter(self, *a, **k):
                return _Counter()

            def create_histogram(self, *a, **k):
                return _Histogram()

        return _Meter()


try:
    from analytics.metrics_store import record_metric, TS_PATH
except Exception:  # pragma: no cover - fallback if analytics not installed
    TS_PATH = Path(".")

    def record_metric(*a, **k):  # type: ignore
        return None


from analysis.data_quality import apply_quality_checks, check_recency
from analysis.domain_adapter import DomainAdapter
from analysis import tick_anomaly_detector, pipeline_anomaly
from analysis.broker_tca import broker_tca
from data.live_recorder import LiveRecorder
from training.curriculum import build_strategy_curriculum
from model_registry import register_policy, get_policy_path, save_model
import train_online
import train_rl

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
exec_engine = ExecutionEngine()


def run_rl_curriculum(data_path: Path, model_dir: Path) -> None:
    """Train RL policies with a simple → graph strategy curriculum."""

    def _stage(name: str, strategy: str) -> Callable[[], float]:
        def _run() -> float:
            metric = train_rl.launch(
                {
                    "data_path": str(data_path),
                    "model_dir": str(model_dir),
                    "strategy": strategy,
                }
            )
            path = get_policy_path()
            if path:
                register_policy(f"realtime_{name}", path, {"stage": name})
            return float(metric)

        return _run

    scheduler = build_strategy_curriculum(
        _stage("simple", "basic"),
        _stage("combo", "combo"),
        _stage("graph", "graph"),
    )
    scheduler.run()


def run_meta_update(data_path: Path, model_dir: Path) -> None:
    """Persist a meta-learned initialisation for downstream training."""

    try:
        save_model(
            "meta_init",
            {"data_path": str(data_path)},
            {"stage": "meta"},
            model_dir / "meta_init.pkl",
        )
    except Exception:
        pass


def _ensure_conn_mgr() -> None:
    """Initialise the connection manager if it hasn't been already."""
    if getattr(conn_mgr, "_manager", None) is None:
        conn_mgr.init([mt5_direct])


def _ensure_data_downloaded(cfg: dict, root: Path) -> None:
    """Download required datasets for configured symbols if missing."""
    try:
        from data.history import load_history_config
    except Exception:
        return
    symbols = cfg.get("symbols") or [cfg.get("symbol", "EURUSD")]
    for sym in symbols:
        pq_path = root / "data" / f"{sym}_history.parquet"
        if not pq_path.exists():
            try:
                load_history_config(sym, cfg, root, validate=cfg.get("validate", False))
            except Exception as e:
                logger.warning("Failed to download history for %s: %s", sym, e)


from user_risk_inputs import configure_user_risk

if os.getenv("SKIP_USER_RISK_PROMPT", "0") != "1":
    configure_user_risk(sys.argv[1:])

try:  # pragma: no cover - orchestrator may be stubbed in tests
    from core.orchestrator import Orchestrator

    orchestrator = Orchestrator.start()
    watchdog = orchestrator.monitor
except Exception:  # pragma: no cover
    import types

    orchestrator = None
    watchdog = types.SimpleNamespace(
        max_rss_mb=None, max_cpu_pct=None, capabilities=None
    )

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
    await graceful_exit()


async def fetch_ticks(symbol: str, n: int = 1000, retries: int = 3) -> pd.DataFrame:
    """Fetch recent tick data from the active broker asynchronously."""
    cfg = load_config()
    max_empty = cfg.get("max_empty_batches", 3)
    global _empty_batch_count
    with tracer.start_as_current_span("fetch_ticks"):
        for attempt in range(retries):
            _ensure_conn_mgr()
            broker = conn_mgr.get_active_broker()
            ticks = await asyncio.to_thread(
                broker.copy_ticks_from,
                symbol,
                int(time.time()) - n,
                n,
                broker.COPY_TICKS_ALL,
            )
            if ticks is None:
                logger.warning(
                    "Failed to fetch ticks for %s on attempt %d, attempting failover",
                    symbol,
                    attempt + 1,
                )
                RECONNECT_COUNT.inc()
                conn_mgr.failover()
                await asyncio.sleep(1)
                continue
            if len(ticks) == 0:
                _empty_batch_count += 1
                record_metric(
                    "fetch_empty_batches", _empty_batch_count, tags={"symbol": symbol}
                )
                if _empty_batch_count >= max_empty:
                    logger.error(
                        "No ticks received for %s after %d empty batches",
                        symbol,
                        _empty_batch_count,
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
            df.rename(
                columns={"bid": "Bid", "ask": "Ask", "volume": "Volume"}, inplace=True
            )
            df = df[["Timestamp", "Bid", "Ask", "Volume"]]
            df["BidVolume"] = df["Volume"]
            df["AskVolume"] = df["Volume"]
            df.drop(columns=["Volume"], inplace=True)
            df = sanitize_ticks(df)
            df, report = apply_quality_checks(df)
            if not check_recency(df, max_age="5s"):
                logger.warning("Discarding stale tick batch for %s", symbol)
                return pd.DataFrame()
            if any(report.values()):
                logger.info("Realtime data quality: %s", report)
                for k, v in report.items():
                    record_metric(f"data_quality_{k}", v, tags={"source": "realtime"})
            ticks_before = len(df)
            df, anomalies = tick_anomaly_detector.filter(df, symbol)
            if anomalies:
                rate = anomalies / max(ticks_before, 1)
                record_metric(
                    "tick_anomalies_total", anomalies, tags={"symbol": symbol}
                )
                record_metric("tick_anomaly_rate", rate, tags={"symbol": symbol})
                send_alert(f"{symbol} tick anomaly rate {rate:.2%}")
            if not pipeline_anomaly.validate(df):
                logger.warning(
                    "Pipeline anomaly detected for %s ticks; dropping batch", symbol
                )
                return pd.DataFrame()
            _ticks_counter.add(len(df))
            return df
        return pd.DataFrame()


async def generate_features(context: pd.DataFrame) -> pd.DataFrame:
    """Asynchronously generate features from context data."""
    with tracer.start_as_current_span("generate_features"):
        if context.empty:
            return pd.DataFrame()
        df = await asyncio.to_thread(make_features, context)
        for row in df.to_dict(orient="records"):
            record_event("feature", row)
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


async def dispatch_signals(bus, df: pd.DataFrame) -> None:
    """Asynchronously dispatch signals to the message bus."""
    with tracer.start_as_current_span("dispatch_signals"):
        if bus is None or df.empty:
            return
        if not pipeline_anomaly.validate(df):
            logger.warning("Pipeline anomaly detected in dispatch; dropping batch")
            return
        await publish_dataframe_async(bus, df)

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
                order_ts = pd.Timestamp.utcnow()
                start = time.perf_counter()
                place_order(symbol=symbol, side=side, volume=size)
                latency = time.perf_counter() - start
                fill_ts = order_ts + pd.to_timedelta(latency, unit="s")
                price = ask if side.lower() == "buy" else bid
                sign = 1 if side.lower() == "buy" else -1
                realized = (price - mid) / mid * sign * 10000.0 if mid else 0.0
                try:
                    broker = conn_mgr.get_active_broker()
                    name = getattr(broker, "__name__", broker.__class__.__name__)
                    broker_tca.record(name, order_ts, fill_ts, realized)
                except Exception:
                    pass


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
            fetch_results = await asyncio.gather(
                *(fetch_fn(sym, 500) for sym in symbols)
            )
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
    root = Path(__file__).resolve().parent
    recorder = LiveRecorder(
        root / "data" / "live",
        batch_size=cfg.get("recorder_batch_size", 500),
        flush_interval=cfg.get("recorder_flush_interval", 1.0),
    )
    _ensure_data_downloaded(cfg, root)
    input("Please log into your MetaTrader 5 terminal and press Enter to continue...")
    while not mt5_direct.is_terminal_logged_in():
        input("MetaTrader 5 terminal not logged in. Log in and press Enter to retry...")
    conn_mgr.init([mt5_direct])
    with tracer.start_as_current_span("train_realtime"):
        if watchdog.max_rss_mb or watchdog.max_cpu_pct:
            watchdog.alert_callback = lambda msg: asyncio.create_task(
                _handle_resource_breach(msg)
            )
        seed = cfg.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        scaler_path = root / "scaler.pkl"
        scaler = FeatureScaler.load(scaler_path)
        adapter_path = root / "domain_adapter.pkl"
        adapter = DomainAdapter.load(adapter_path)
        batch_count = 0
        reestimate_interval = cfg.get("adapter_reestimate_interval", 100)
        rl_update_interval = cfg.get("rl_update_interval", 1000)
        meta_update_interval = cfg.get("meta_update_interval", 5000)

        symbols = cfg.get("symbols") or [cfg.get("symbol", "EURUSD")]

        signal_backend = cfg.get("signal_backend", "none")
        bus = None
        if signal_backend != "none":
            bus = get_signal_backend(cfg)
            logger.info("Using %s backend for signal bus", signal_backend)

        async def process_batch(batch: pd.DataFrame) -> None:
            nonlocal batch_count
            feats = await generate_features(batch)
            if feats.empty:
                return
            feats = apply_liquidity_adjustment(feats)
            num_cols = feats.select_dtypes(np.number).columns
            if len(num_cols) > 0:
                raw_feats = feats[num_cols].copy()
                feats[num_cols] = adapter.transform(feats[num_cols])
                scaler.partial_fit(feats[num_cols])
                feats[num_cols] = scaler.transform(feats[num_cols])
                scaler.save(scaler_path)
                batch_count += 1
                if batch_count % reestimate_interval == 0:
                    adapter.reestimate(raw_feats)
                    adapter.save(adapter_path)
                    if (
                        adapter.source_mean_ is not None
                        and adapter.target_mean_ is not None
                        and adapter.source_cov_ is not None
                        and adapter.target_cov_ is not None
                    ):
                        mean_diff = float(
                            np.linalg.norm(adapter.source_mean_ - adapter.target_mean_)
                        )
                        cov_diff = float(
                            np.linalg.norm(adapter.source_cov_ - adapter.target_cov_)
                        )
                        logger.info(
                            "Domain adapter alignment: mean diff %.4f cov diff %.4f",
                            mean_diff,
                            cov_diff,
                        )
            if not pipeline_anomaly.validate(feats):
                logger.warning("Pipeline anomaly detected in features; dropping batch")
                return
            await dispatch_signals(bus, feats)
            await asyncio.to_thread(
                train_online.train_online,
                data_path=recorder.root,
                model_dir=root / "models",
                min_ticks=cfg.get("online_min_ticks", 1000),
                interval=cfg.get("online_interval", 300),
                run_once=True,
            )
            # Schedule periodic RL and meta-learning updates based on the
            # number of processed batches.  Each update trains on ticks
            # recorded by ``LiveRecorder`` and persists the resulting policies
            # through :mod:`model_registry` for later reuse.
            if batch_count and batch_count % rl_update_interval == 0:
                await asyncio.to_thread(
                    run_rl_curriculum, recorder.root, root / "models"
                )
            if batch_count and batch_count % meta_update_interval == 0:
                await asyncio.to_thread(run_meta_update, recorder.root, root / "models")

        tick_queue: asyncio.Queue = asyncio.Queue()
        train_queue: asyncio.Queue = asyncio.Queue()
        producer = asyncio.create_task(
            tick_producer(
                symbols,
                tick_queue,
                throttle_threshold=cfg.get("backlog_threshold", 1000),
            )
        )
        recorder_task = asyncio.create_task(recorder.run(tick_queue, train_queue))
        workers = [
            asyncio.create_task(
                tick_worker(
                    train_queue,
                    process_batch,
                    target_latency=cfg.get("target_batch_latency", 1.0),
                )
            )
            for _ in range(cfg.get("worker_count", 1))
        ]
        await asyncio.gather(producer, recorder_task, *workers)


if __name__ == "__main__":
    asyncio.run(train_realtime())
