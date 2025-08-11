import asyncio
import time
from pathlib import Path
import os
from collections import deque
import random
import logging
import numpy as np
import pandas as pd
import joblib
from git import Repo

from brokers import connection_manager as conn_mgr
import MetaTrader5 as mt5  # type: ignore

from utils import load_config
try:  # allow tests to stub out utils
    from utils.resource_monitor import ResourceMonitor
except Exception:  # pragma: no cover - fallback when utils is stubbed
    class ResourceMonitor:  # type: ignore
        def __init__(self, *a, **k):
            self.max_rss_mb = None
            self.max_cpu_pct = None

        def start(self) -> None:  # noqa: D401 - trivial
            """No-op start when resource monitor is unavailable."""
            return
from data.features import make_features
import duckdb
from log_utils import setup_logging, log_exceptions
from analysis.anomaly_detector import detect_anomalies
from metrics import RECONNECT_COUNT, FEATURE_ANOMALIES, RESOURCE_RESTARTS
from signal_queue import get_signal_backend
from data.sanitize import sanitize_ticks
from models import model_store
from data.trade_log import TradeLog
from event_store import EventStore

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
    repo_path = Path(__file__).resolve().parent
    db_path = repo_path / "data" / "realtime.duckdb"
    model_path = repo_path / "model.joblib"
    trade_log = TradeLog(repo_path / "data" / "trade_log.db")
    quarantine_path = repo_path / "data" / "anomaly_quarantine.csv"
    event_store = EventStore(repo_path / "data" / "events.db")

    version_id = os.getenv("MODEL_VERSION_ID")
    if version_id:
        try:
            model, _ = model_store.load_model(version_id)
            joblib.dump(model, model_path)
            logger.info("Loaded model version %s", version_id)
        except FileNotFoundError:
            logger.warning("Model version %s not found", version_id)

    window = cfg.get("realtime_window", 10000)
    context_rows = 300  # number of rows to keep for feature context

    repo = Repo(repo_path)

    broker = conn_mgr.get_active_broker()
    positions_get = getattr(broker, "positions_get", lambda: [])
    broker_positions = await asyncio.to_thread(positions_get)
    trade_log.sync_mt5_positions(broker_positions)

    symbols = cfg.get("symbols") or [cfg.get("symbol", "EURUSD")]

    signal_backend = cfg.get("signal_backend", "zmq")
    queue = None
    if signal_backend in {"kafka", "redis"}:
        queue = get_signal_backend(cfg)
        logger.info("Using %s backend for signal queue", signal_backend)

    conn = duckdb.connect(db_path.as_posix())

    # rolling buffers holding recent tick data for feature calculations
    tick_buffers = {sym: deque(maxlen=context_rows) for sym in symbols}

    if conn.execute("SELECT table_name FROM information_schema.tables WHERE table_name='history'").fetchone():
        history = conn.execute(
            f"""
            SELECT * FROM (
                SELECT *, row_number() OVER (PARTITION BY Symbol ORDER BY Timestamp DESC) AS rn
                FROM history
            ) WHERE rn <= {window}
            ORDER BY Timestamp
            """
        ).fetch_df()
    else:
        history = pd.DataFrame(columns=["Timestamp", "Bid", "Ask", "BidVolume", "AskVolume", "Symbol"])

    if conn.execute("SELECT table_name FROM information_schema.tables WHERE table_name='features'").fetchone():
        feature_df = conn.execute(
            f"""
            SELECT * FROM (
                SELECT *, row_number() OVER (PARTITION BY Symbol ORDER BY Timestamp DESC) AS rn
                FROM features
            ) WHERE rn <= {window}
            ORDER BY Timestamp
            """
        ).fetch_df()
    else:
        feature_df = make_features(history) if not history.empty else pd.DataFrame()

    # seed buffers with existing history
    if not history.empty:
        for sym in symbols:
            sym_hist = history[history["Symbol"] == sym].tail(context_rows)
            for rec in sym_hist.to_dict("records"):
                tick_buffers[sym].append(rec)

    empty_iters = 0
    while True:
        tick_frames = []
        fetch_results = await asyncio.gather(
            *(fetch_ticks(sym, 500) for sym in symbols)
        )
        for sym, ticks in zip(symbols, fetch_results):
            if not ticks.empty:
                ticks["Symbol"] = sym
                tick_frames.append(ticks)
                # update rolling buffer for this symbol
                for rec in ticks.to_dict("records"):
                    # avoid duplicate timestamps in the buffer
                    if not tick_buffers[sym] or tick_buffers[sym][-1]["Timestamp"] != rec["Timestamp"]:
                        tick_buffers[sym].append(rec)

        if not tick_frames:
            empty_iters += 1
            if empty_iters >= 3:
                backoff = min(300, 60 * empty_iters)
                logger.warning("No ticks received for %d iterations, attempting failover", empty_iters)
                RECONNECT_COUNT.inc()
                conn_mgr.failover()
                await asyncio.sleep(backoff)
                empty_iters = 0
            else:
                await asyncio.sleep(60)
            continue
        else:
            empty_iters = 0

        new_ticks = pd.concat(tick_frames, ignore_index=True)
        new_ticks["Timestamp"] = pd.to_datetime(new_ticks["Timestamp"])

        history = (
            pd.concat([history, new_ticks])
            .drop_duplicates(subset=["Timestamp", "Symbol"], keep="last")
            .sort_values("Timestamp")
        )
        history = history.groupby("Symbol", as_index=False, group_keys=False).tail(window)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, conn.register, "new_ticks", new_ticks)
        await loop.run_in_executor(
            None,
            conn.execute,
            "CREATE TABLE IF NOT EXISTS history AS SELECT * FROM new_ticks LIMIT 0",
        )
        await loop.run_in_executor(None, conn.execute, "INSERT INTO history SELECT * FROM new_ticks")
        await loop.run_in_executor(
            None,
            conn.execute,
            f"""
            DELETE FROM history
            WHERE rowid IN (
                SELECT rowid FROM (
                    SELECT rowid,
                           row_number() OVER (PARTITION BY Symbol ORDER BY Timestamp DESC) AS rn
                    FROM history
                ) t WHERE rn > {window}
            )
            """,
        )

        # build context from rolling buffers for all symbols
        context_frames = [pd.DataFrame(list(buf)) for buf in tick_buffers.values()]
        context = pd.concat(context_frames, ignore_index=True)
        context = context.drop_duplicates(subset=["Timestamp", "Symbol"], keep="last")

        df = await generate_features(context)
        new_features = df.merge(
            new_ticks[["Timestamp", "Symbol"]], on=["Timestamp", "Symbol"], how="inner"
        )
        new_features, anomalies = detect_anomalies(
            new_features, quarantine_path=quarantine_path, counter=FEATURE_ANOMALIES
        )
        for rec in new_features.to_dict("records"):
            rec = {**rec, "Timestamp": str(rec.get("Timestamp"))}
            event_store.record("feature", rec)
        feature_df = (
            pd.concat([feature_df, new_features])
            .drop_duplicates(subset=["Timestamp", "Symbol"], keep="last")
            .sort_values("Timestamp")
        )
        feature_df = feature_df.groupby("Symbol", as_index=False, group_keys=False).tail(window)

        await loop.run_in_executor(None, conn.register, "new_feat", new_features)
        await loop.run_in_executor(
            None,
            conn.execute,
            "CREATE TABLE IF NOT EXISTS features AS SELECT * FROM new_feat LIMIT 0",
        )
        await loop.run_in_executor(None, conn.execute, "INSERT INTO features SELECT * FROM new_feat")
        await loop.run_in_executor(
            None,
            conn.execute,
            f"""
            DELETE FROM features
            WHERE rowid IN (
                SELECT rowid FROM (
                    SELECT rowid,
                           row_number() OVER (PARTITION BY Symbol ORDER BY Timestamp DESC) AS rn
                    FROM features
                ) t WHERE rn > {window}
            )
            """,
        )

        if feature_df.empty:
            await asyncio.sleep(60)
            continue

        df = feature_df.copy()
        if "Symbol" in df.columns:
            df["SymbolCode"] = df["Symbol"].astype("category").cat.codes

        # base set of features used for model training
        features = [
            "return",
            "ma_5",
            "ma_10",
            "ma_30",
            "ma_60",
            "volatility_30",
            "spread",
            "rsi_14",
            "market_regime",
        ]
        # include optional features when present
        if "news_sentiment" in df.columns:
            features.append("news_sentiment")
        features.extend(
            [
                c
                for c in df.columns
                if c.startswith("cross_corr_")
                or c.startswith("factor_")
                or c.startswith("cross_mom_")
            ]
        )
        if "volume_ratio" in df.columns:
            features.extend(["volume_ratio", "volume_imbalance"])
        if "SymbolCode" in df.columns:
            features.append("SymbolCode")
        X = df[features]
        y = (df["return"].shift(-1) > 0).astype(int)

        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from lightgbm import LGBMClassifier

        steps = []
        if cfg.get("use_scaler", True):
            steps.append(("scaler", StandardScaler()))
        steps.append(("clf", LGBMClassifier(n_estimators=200, random_state=42)))
        pipe = Pipeline(steps)

        await asyncio.to_thread(pipe.fit, X, y)
        joblib.dump(pipe, model_path)

        # dispatch model predictions for the newly generated features
        probs = await asyncio.to_thread(pipe.predict_proba, new_features[features])
        out = new_features[["Timestamp", "Symbol"]].copy()
        out["prob"] = probs[:, 1]
        await dispatch_signals(queue, out)
        for rec in out.to_dict("records"):
            event_store.record(
                "prediction", {**rec, "Timestamp": str(rec.get("Timestamp"))}
            )
            sym_ticks = new_ticks[new_ticks["Symbol"] == rec["Symbol"]]
            if sym_ticks.empty:
                continue
            price = float(sym_ticks["Bid"].iloc[-1])
            side = "BUY" if rec["prob"] > 0.5 else "SELL"
            order = {
                "timestamp": rec["Timestamp"],
                "symbol": rec["Symbol"],
                "side": side,
                "volume": 1.0,
                "price": price,
            }
            order_id = trade_log.record_order(order)
            event_store.record("order", {**order, "order_id": order_id, "timestamp": str(order["timestamp"])})
            trade_log.record_fill({**order, "order_id": order_id})
            event_store.record(
                "fill",
                {**order, "order_id": order_id, "timestamp": str(order["timestamp"])},
            )

        returns = df["return"].dropna()
        from backtest import compute_metrics
        metrics = compute_metrics(returns)
        equity = (1 + returns).cumprod().tolist()
        try:
            import requests
            api_url = os.getenv("API_URL", "http://localhost:8000/metrics")
            headers = {"x-api-key": os.getenv("API_KEY", "")}
            requests.post(api_url, json={"equity_curve": equity, "metrics": metrics}, headers=headers, timeout=5)
        except Exception as exc:
            logger.error("Failed to broadcast metrics: %s", exc)

        # commit updates
        await loop.run_in_executor(None, conn.execute, "checkpoint")
        repo.git.add(db_path.as_posix())
        repo.git.add(model_path.as_posix())
        repo.index.commit("Update model with realtime data")
        try:
            repo.remote().push()
        except Exception as e:
            logger.info("Git push failed: %s", e)

        await asyncio.sleep(300)


if __name__ == "__main__":
    asyncio.run(train_realtime())
