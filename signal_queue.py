from __future__ import annotations

import os
import json
import time
import pandas as pd
import zmq
import zmq.asyncio
from typing import AsyncGenerator, Any
from contextlib import contextmanager, asynccontextmanager

from kafka import KafkaProducer, KafkaConsumer, TopicPartition, errors as kafka_errors
import redis

from analytics.metrics_store import record_metric
from analytics import decision_logger
from risk.position_sizer import PositionSizer
from models import conformal
from models.quantile_regression import GradientBoostedQuantile, NeuralQuantile  # type: ignore
from analysis.quantile_forecast import log_quantile_forecast
from strategies.trade_exit_policy import TradeExitPolicy
from analysis.entry_value import EntryValueScorer, log_entry_value

from telemetry import get_tracer, get_meter
from strategy import StrategyRouter
from news.impact_model import get_impact

tracer = get_tracer(__name__)
meter = get_meter(__name__)
_publish_count = meter.create_counter(
    "signals_published", description="Number of signals published"
)
_consume_count = meter.create_counter(
    "signals_consumed", description="Number of signals consumed"
)

_MAX_INTERVAL_WIDTH = float(os.getenv("CONFORMAL_MAX_WIDTH", "inf"))
_DROP_WIDE_INTERVALS = os.getenv("CONFORMAL_DROP_WIDE", "0") == "1"
_CONFORMAL_Q = float(os.getenv("CONFORMAL_RESIDUAL_Q", "0.0"))

_IMPACT_THRESHOLD = float(os.getenv("NEWS_IMPACT_THRESHOLD", "0.5"))
_UNCERTAINTY_THRESHOLD = float(os.getenv("NEWS_IMPACT_UNCERTAINTY", "1.0"))
_IMPACT_BOOST = float(os.getenv("NEWS_IMPACT_BOOST", "1.5"))
_DOWNSCALE = float(os.getenv("NEWS_IMPACT_DOWNSCALE", "0.5"))

# Lazy registry to avoid heavy initialization at import time
_REGISTRY = None

# Global strategy router that adapts allocations between algorithms based on
# simple regime features.  It is intentionally lightweight so importing this
# module does not pull in heavy dependencies.
_ROUTER = StrategyRouter()


def _get_registry():  # pragma: no cover - simple accessor
    global _REGISTRY
    if _REGISTRY is None:
        from model_registry import ModelRegistry

        _REGISTRY = ModelRegistry(auto_refresh=False)
    return _REGISTRY


_QUANTILE_MODEL_ID = os.getenv("QUANTILE_MODEL_ID")
_QUANTILE_MODEL = None


def _get_quantile_model():
    """Load cached quantile model if configured."""
    global _QUANTILE_MODEL
    if _QUANTILE_MODEL is None and _QUANTILE_MODEL_ID:
        try:
            from models import model_store
            _QUANTILE_MODEL, _ = model_store.load_model(_QUANTILE_MODEL_ID)
        except Exception:
            _QUANTILE_MODEL = None
    return _QUANTILE_MODEL


def _predict_var_es(prob: float, conf: float) -> tuple[float | None, float | None]:
    model = _get_quantile_model()
    if model is None:
        return None, None
    try:
        X = pd.DataFrame([[prob, conf]], columns=["prob", "confidence"]).to_numpy()
        if hasattr(model, "var_es"):
            var, es = model.var_es(X, 0.05)
            v = float(var[0]) if var is not None else None
            e = float(es[0]) if es is not None else None
            return v, e
        preds = model.predict(X)
        var = preds.get(0.05)
        return (float(var[0]) if var is not None else None, None)
    except Exception:
        return None, None
import asyncio
from proto import signals_pb2
from event_store import EventStore
from pathlib import Path

_EVENT_STORE = EventStore(Path(os.getenv("EVENT_STORE_PATH", Path(__file__).resolve().parent / "data" / "events.db")))

_CTX = zmq.Context.instance()
_ASYNC_CTX = zmq.asyncio.Context.instance()

_QUEUE_DEPTH = 0


def update_router(features: dict, reward: float, algorithm: str) -> None:
    """Update the global strategy router with a realised reward."""
    _ROUTER.update(features, reward, algorithm)


def _passes_meta(meta_clf: Any | None, prob: float, conf: float) -> bool:
    """Return ``True`` if a message passes the meta-classifier filter.

    ``meta_clf`` may be a model instance or a string identifying a model in
    :mod:`model_registry`.  When a string is supplied and the optimal variant
    requires more resources than available locally, predictions are delegated to
    the remote model server via :mod:`models.remote_client`.
    """

    if meta_clf is None:
        return True
    feats = pd.DataFrame([[prob, conf]], columns=["prob", "confidence"])
    try:
        if isinstance(meta_clf, str):
            registry = _get_registry()
            from models.remote_client import predict_remote

            if registry.requires_remote(meta_clf):
                preds = predict_remote(meta_clf, feats)
                return bool(preds[0]) if preds else True
            # attempt local load
            try:  # pragma: no cover - optional dependency path
                from models import model_store

                model, _ = model_store.load_model(meta_clf)
                return bool(model.predict(feats)[0])
            except Exception:
                preds = predict_remote(meta_clf, feats)
                return bool(preds[0]) if preds else True
        return bool(meta_clf.predict(feats)[0])
    except Exception:  # pragma: no cover - safeguard against bad meta_clf
        return True

@contextmanager
def get_publisher(bind_address: str | None = None) -> zmq.Socket:
    """Return a PUB socket bound to the given address."""
    addr = bind_address or os.getenv("SIGNAL_QUEUE_BIND", "tcp://*:5555")
    sock = _CTX.socket(zmq.PUB)
    sock.bind(addr)
    try:
        yield sock
    finally:
        sock.close()


@contextmanager
def get_subscriber(connect_address: str | None = None, topic: str = "") -> zmq.Socket:
    """Return a SUB socket connected to the given address."""
    addr = connect_address or os.getenv("SIGNAL_QUEUE_URL", "tcp://localhost:5555")
    sock = _CTX.socket(zmq.SUB)
    sock.connect(addr)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    try:
        yield sock
    finally:
        sock.close()


@asynccontextmanager
async def get_async_publisher(bind_address: str | None = None) -> zmq.asyncio.Socket:
    """Return an async PUB socket bound to the given address."""
    addr = bind_address or os.getenv("SIGNAL_QUEUE_BIND", "tcp://*:5555")
    sock = _ASYNC_CTX.socket(zmq.PUB)
    sock.bind(addr)
    try:
        yield sock
    finally:
        sock.close()


@asynccontextmanager
async def get_async_subscriber(connect_address: str | None = None, topic: str = "") -> zmq.asyncio.Socket:
    """Return an async SUB socket connected to the given address."""
    addr = connect_address or os.getenv("SIGNAL_QUEUE_URL", "tcp://localhost:5555")
    sock = _ASYNC_CTX.socket(zmq.SUB)
    sock.connect(addr)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    try:
        yield sock
    finally:
        sock.close()


def _encode_field(field: int, value: str) -> bytes:
    """Encode a simple length-delimited field for ZeroMQ messages."""
    data = value.encode("utf-8")
    key = (field << 3) | 2
    out = bytearray([key])
    length = len(data)
    while True:
        b = length & 0x7F
        length >>= 7
        out.append(b | (0x80 if length else 0))
        if not length:
            break
    out.extend(data)
    return bytes(out)


def request_history(
    sock: zmq.Socket,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    path: str = "history_request.csv",
) -> pd.DataFrame:
    """Send a history request command and return the resulting dataframe.

    Parameters
    ----------
    sock: zmq.Socket
        ZeroMQ publisher socket used for sending the request.
    symbol: str
        Instrument symbol to fetch.
    timeframe: str
        Timeframe string such as ``"M1"`` or ``"H1"``.
    start: str
        Start timestamp in ``YYYY.MM.DD HH:MM`` format.
    end: str
        End timestamp in the same format.
    path: str, optional
        Location where the EA will write the resulting CSV file.
    """

    msg = b"".join(
        [
            _encode_field(10, "history_request"),
            _encode_field(11, symbol),
            _encode_field(12, timeframe),
            _encode_field(13, start),
            _encode_field(14, end),
            _encode_field(15, path),
        ]
    )
    sock.send(msg)
    file_path = Path(path)
    for _ in range(100):
        if file_path.exists():
            return pd.read_csv(file_path)
        time.sleep(0.1)
    raise FileNotFoundError(path)


def publish_dataframe(sock: zmq.Socket, df: pd.DataFrame, fmt: str = "protobuf") -> None:
    """Publish rows of a dataframe as JSON or Protobuf messages."""
    global _QUEUE_DEPTH
    algos: list[str] = []
    sizes: list[float] = []
    with tracer.start_as_current_span("publish_dataframe"):
        fmt = fmt.lower()
        for _, row in df.iterrows():
            _QUEUE_DEPTH += 1
            try:
                record_metric("queue_depth", _QUEUE_DEPTH)
            except Exception:
                pass
            feats = {
                "volatility": float(row.get("volatility_30", 0.0)),
                "trend_strength": float(row.get("trend_strength", 0.0)),
            }
            algo, action = _ROUTER.act(feats)
            algos.append(algo)
            sizes.append(float(action))
            if fmt == "json":
                payload = {
                    "Timestamp": str(row["Timestamp"]),
                    "Symbol": str(row.get("Symbol", "")),
                    "prob": float(row["prob"]),
                    "confidence": float(row.get("confidence", 1.0)),
                }
                if "var" in row:
                    payload["var"] = float(row["var"])
                if "es" in row:
                    payload["es"] = float(row["es"])
                for field in [
                    "depth_imbalance",
                    "volatility_30",
                    "regime_embed",
                    "future_return",
                    "total_depth",
                ]:
                    if field in row:
                        payload[field] = row[field]
                sock.send_json(payload)
            else:
                msg = signals_pb2.Signal(
                    timestamp=str(row["Timestamp"]),
                    symbol=str(row.get("Symbol", "")),
                    probability=str(row["prob"]),
                    confidence=float(row.get("confidence", 1.0)),
                )
                sock.send(msg.SerializeToString())
            _publish_count.add(1)
    df_log = df.copy()
    if "Timestamp" in df_log.columns:
        df_log = df_log.rename(columns={"Timestamp": "timestamp"})
    df_log["algorithm"] = algos
    df_log["position_size"] = sizes
    decision_logger.log(df_log)


async def publish_dataframe_async(
    sock: zmq.asyncio.Socket, df: pd.DataFrame, fmt: str = "protobuf"
) -> None:
    """Asynchronously publish rows of a dataframe as JSON or Protobuf messages."""
    global _QUEUE_DEPTH
    algos: list[str] = []
    sizes: list[float] = []
    with tracer.start_as_current_span("publish_dataframe_async"):
        fmt = fmt.lower()
        for _, row in df.iterrows():
            _QUEUE_DEPTH += 1
            try:
                record_metric("queue_depth", _QUEUE_DEPTH)
            except Exception:
                pass
            feats = {
                "volatility": float(row.get("volatility_30", 0.0)),
                "trend_strength": float(row.get("trend_strength", 0.0)),
            }
            algo, action = _ROUTER.act(feats)
            algos.append(algo)
            sizes.append(float(action))
            if fmt == "json":
                payload = {
                    "Timestamp": str(row["Timestamp"]),
                    "Symbol": str(row.get("Symbol", "")),
                    "prob": float(row["prob"]),
                    "confidence": float(row.get("confidence", 1.0)),
                }
                if "var" in row:
                    payload["var"] = float(row["var"])
                if "es" in row:
                    payload["es"] = float(row["es"])
                for field in [
                    "depth_imbalance",
                    "volatility_30",
                    "regime_embed",
                    "future_return",
                    "total_depth",
                ]:
                    if field in row:
                        payload[field] = row[field]
                await sock.send_json(payload)
            else:
                msg = signals_pb2.Signal(
                    timestamp=str(row["Timestamp"]),
                    symbol=str(row.get("Symbol", "")),
                    probability=str(row["prob"]),
                    confidence=float(row.get("confidence", 1.0)),
                )
                await sock.send(msg.SerializeToString())
            _publish_count.add(1)
    df_log = df.copy()
    if "Timestamp" in df_log.columns:
        df_log = df_log.rename(columns={"Timestamp": "timestamp"})
    df_log["algorithm"] = algos
    df_log["position_size"] = sizes
    decision_logger.log(df_log)


async def iter_messages(
    sock: zmq.asyncio.Socket,
    fmt: str = "protobuf",
    sizer: PositionSizer | None = None,
    meta_clf: Any | None = None,
    exit_policy: TradeExitPolicy | None = None,
    entry_scorer: EntryValueScorer | None = None,
) -> AsyncGenerator[dict, None]:
    """Yield decoded messages from a subscriber socket as they arrive."""
    global _QUEUE_DEPTH
    fmt = fmt.lower()
    while True:
        with tracer.start_as_current_span("iter_message"):
            if fmt == "json":
                data = await sock.recv_json()
                _QUEUE_DEPTH = max(0, _QUEUE_DEPTH - 1)
                try:
                    record_metric("queue_depth", _QUEUE_DEPTH)
                except Exception:
                    pass
                symbol = data.get("Symbol", "")
                prob = float(data.get("prob", 0.0))
                conf = float(data.get("confidence", 1.0))
                var = data.get("var")
                es = data.get("es")
                if var is None or es is None:
                    q_var, q_es = _predict_var_es(prob, conf)
                    if var is None:
                        var = q_var
                    if es is None:
                        es = q_es
                lower, upper = conformal.predict_interval([prob], _CONFORMAL_Q)
                width = float(upper[0] - lower[0])
                if width > _MAX_INTERVAL_WIDTH:
                    if _DROP_WIDE_INTERVALS:
                        continue
                    scale = _MAX_INTERVAL_WIDTH / width if _MAX_INTERVAL_WIDTH != float("inf") else 0.0
                else:
                    scale = 1.0
                base_size = (
                    sizer.size(prob, symbol, var=var, es=es, confidence=conf)
                    if sizer
                    else prob * conf
                )
                size = base_size * scale
                impact, uncert = get_impact(symbol, data.get("Timestamp", ""))
                direction = 1 if prob >= 0.5 else -1
                if uncert > _UNCERTAINTY_THRESHOLD:
                    size *= _DOWNSCALE
                if impact is not None and abs(impact) > _IMPACT_THRESHOLD:
                    if impact * direction < 0:
                        continue
                    size *= _IMPACT_BOOST
                payload = {
                    "Timestamp": data.get("Timestamp", ""),
                    "Symbol": symbol,
                    "prob": prob,
                    "confidence": conf,
                    "var": var,
                    "es": es,
                    "size": size,
                    "impact": impact,
                    "uncertainty": uncert,
                    "interval": (float(lower[0]), float(upper[0])),
                    "interval_width": width,
                }
                try:
                    realised = None
                    try:
                        realised = data.get("future_return")  # type: ignore[union-attr]
                    except Exception:
                        pass
                    if realised is None:
                        try:
                            realised = getattr(sig, "future_return")  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    if realised is None:
                        try:
                            realised = payload.get("future_return")  # type: ignore[union-attr]
                        except Exception:
                            pass
                    log_quantile_forecast(payload.get("Timestamp", ""), symbol, 0.05, var, None if realised is None else float(realised))
                except Exception:
                    pass
                if entry_scorer is not None:
                    depth = float(data.get("depth_imbalance", 0.0))
                    vol = float(data.get("volatility_30", 0.0))
                    embed = data.get("regime_embed", [])
                    score = entry_scorer.score(depth, vol, embed)
                    payload["expected_value"] = score
                    realised = float(data.get("future_return", 0.0))
                    try:
                        log_entry_value(payload.get("Timestamp", ""), symbol, score, realised)
                    except Exception:
                        pass
                    if score <= 0:
                        continue
                if not _passes_meta(meta_clf, prob, conf):
                    continue
                _EVENT_STORE.record("prediction", payload)
                _consume_count.add(1)
                yield payload
                if exit_policy and exit_policy.should_exit(symbol, payload):
                    close_msg = {"action": "close", "Symbol": symbol, "Timestamp": payload.get("Timestamp")}
                    _EVENT_STORE.record("exit", close_msg)
                    yield close_msg
            else:
                raw = await sock.recv()
                sig = signals_pb2.Signal()
                sig.ParseFromString(raw)
                _QUEUE_DEPTH = max(0, _QUEUE_DEPTH - 1)
                try:
                    record_metric("queue_depth", _QUEUE_DEPTH)
                except Exception:
                    pass
                prob = float(sig.probability)
                symbol = sig.symbol
                conf = float(getattr(sig, "confidence", 1.0))
                var = getattr(sig, "var", None)
                es = getattr(sig, "es", None)
                if var is None or es is None:
                    q_var, q_es = _predict_var_es(prob, conf)
                    if var is None:
                        var = q_var
                    if es is None:
                        es = q_es
                if var is not None:
                    var = float(var)
                if es is not None:
                    es = float(es)
                lower, upper = conformal.predict_interval([prob], _CONFORMAL_Q)
                width = float(upper[0] - lower[0])
                if width > _MAX_INTERVAL_WIDTH:
                    if _DROP_WIDE_INTERVALS:
                        continue
                    scale = _MAX_INTERVAL_WIDTH / width if _MAX_INTERVAL_WIDTH != float("inf") else 0.0
                else:
                    scale = 1.0
                base_size = (
                    sizer.size(prob, symbol, var=var, es=es, confidence=conf)
                    if sizer
                    else prob * conf
                )
                size = base_size * scale
                impact, uncert = get_impact(symbol, sig.timestamp)
                direction = 1 if prob >= 0.5 else -1
                if uncert > _UNCERTAINTY_THRESHOLD:
                    size *= _DOWNSCALE
                if impact is not None and abs(impact) > _IMPACT_THRESHOLD:
                    if impact * direction < 0:
                        continue
                    size *= _IMPACT_BOOST
                payload = {
                    "Timestamp": sig.timestamp,
                    "Symbol": symbol,
                    "prob": prob,
                    "confidence": conf,
                    "var": var,
                    "es": es,
                    "size": size,
                    "impact": impact,
                    "uncertainty": uncert,
                    "interval": (float(lower[0]), float(upper[0])),
                    "interval_width": width,
                }
                try:
                    realised = None
                    try:
                        realised = data.get("future_return")  # type: ignore[union-attr]
                    except Exception:
                        pass
                    if realised is None:
                        try:
                            realised = getattr(sig, "future_return")  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    if realised is None:
                        try:
                            realised = payload.get("future_return")  # type: ignore[union-attr]
                        except Exception:
                            pass
                    log_quantile_forecast(payload.get("Timestamp", ""), symbol, 0.05, var, None if realised is None else float(realised))
                except Exception:
                    pass
                if entry_scorer is not None:
                    depth = float(getattr(sig, "depth_imbalance", 0.0))
                    vol = float(getattr(sig, "volatility_30", 0.0))
                    embed = getattr(sig, "regime_embed", [])
                    score = entry_scorer.score(depth, vol, embed)
                    payload["expected_value"] = score
                    realised = float(getattr(sig, "future_return", 0.0))
                    try:
                        log_entry_value(payload.get("Timestamp", ""), symbol, score, realised)
                    except Exception:
                        pass
                    if score <= 0:
                        continue
                if not _passes_meta(meta_clf, prob, conf):
                    continue
                _EVENT_STORE.record("prediction", payload)
                _consume_count.add(1)
                yield payload
                if exit_policy and exit_policy.should_exit(symbol, payload):
                    close_msg = {"action": "close", "Symbol": symbol, "Timestamp": payload.get("Timestamp")}
                    _EVENT_STORE.record("exit", close_msg)
                    yield close_msg


class KafkaSignalQueue:
    """Durable signal queue backed by Kafka."""

    def __init__(
        self,
        topic: str = "signals",
        bootstrap_servers: str | None = None,
        group_id: str = "signal-queue",
    ) -> None:
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self.group_id = group_id
        self.producer = KafkaProducer(bootstrap_servers=self.bootstrap_servers)
        self.consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
        )
        self.offset = 0

    def close(self) -> None:
        self.producer.close()
        self.consumer.close()

    def publish_dataframe(self, df: pd.DataFrame, fmt: str = "protobuf", retries: int = 3) -> None:
        fmt = fmt.lower()
        algos: list[str] = []
        sizes: list[float] = []
        for _, row in df.iterrows():
            feats = {
                "volatility": float(row.get("volatility_30", 0.0)),
                "trend_strength": float(row.get("trend_strength", 0.0)),
            }
            algo, action = _ROUTER.act(feats)
            algos.append(algo)
            sizes.append(float(action))
            if fmt == "json":
                payload_dict = {
                    "Timestamp": str(row["Timestamp"]),
                    "Symbol": str(row.get("Symbol", "")),
                    "prob": float(row["prob"]),
                    "confidence": float(row.get("confidence", 1.0)),
                }
                if "var" in row:
                    payload_dict["var"] = float(row["var"])
                if "es" in row:
                    payload_dict["es"] = float(row["es"])
                for field in [
                    "depth_imbalance",
                    "volatility_30",
                    "regime_embed",
                    "future_return",
                    "total_depth",
                ]:
                    if field in row:
                        payload_dict[field] = row[field]
                payload = json.dumps(payload_dict).encode()
            else:
                msg = signals_pb2.Signal(
                    timestamp=str(row["Timestamp"]),
                    symbol=str(row.get("Symbol", "")),
                    probability=str(row["prob"]),
                    confidence=float(row.get("confidence", 1.0)),
                )
                payload = msg.SerializeToString()
            for attempt in range(retries):
                try:
                    global _QUEUE_DEPTH
                    _QUEUE_DEPTH += 1
                    try:
                        record_metric("queue_depth", _QUEUE_DEPTH)
                    except Exception:
                        pass
                    future = self.producer.send(self.topic, payload)
                    meta = future.get(timeout=10)
                    self.offset = meta.offset
                    try:
                        record_metric("queue_offset", self.offset)
                    except Exception:
                        pass
                    break
                except kafka_errors.KafkaError:
                    if attempt + 1 == retries:
                        raise
                    time.sleep(1)
        df_log = df.copy()
        if "Timestamp" in df_log.columns:
            df_log = df_log.rename(columns={"Timestamp": "timestamp"})
        df_log["algorithm"] = algos
        df_log["position_size"] = sizes
        decision_logger.log(df_log)

    def iter_messages(
        self,
        fmt: str = "protobuf",
        sizer: PositionSizer | None = None,
        meta_clf: Any | None = None,
        entry_scorer: EntryValueScorer | None = None,
    ):
        fmt = fmt.lower()
        for msg in self.consumer:
            global _QUEUE_DEPTH
            _QUEUE_DEPTH = max(0, _QUEUE_DEPTH - 1)
            try:
                record_metric("queue_depth", _QUEUE_DEPTH)
            except Exception:
                pass
            self.offset = msg.offset
            try:
                record_metric("queue_offset", self.offset)
            except Exception:
                pass
            if fmt == "json":
                data = json.loads(msg.value.decode())
                symbol = data.get("Symbol", "")
                prob = float(data.get("prob", 0.0))
                conf = float(data.get("confidence", 1.0))
                var = data.get("var")
                es = data.get("es")
                if var is None or es is None:
                    q_var, q_es = _predict_var_es(prob, conf)
                    if var is None:
                        var = q_var
                    if es is None:
                        es = q_es
                lower, upper = conformal.predict_interval([prob], _CONFORMAL_Q)
                width = float(upper[0] - lower[0])
                if width > _MAX_INTERVAL_WIDTH:
                    if _DROP_WIDE_INTERVALS:
                        continue
                    scale = _MAX_INTERVAL_WIDTH / width if _MAX_INTERVAL_WIDTH != float("inf") else 0.0
                else:
                    scale = 1.0
                base_size = (
                    sizer.size(prob, symbol, var=var, es=es, confidence=conf)
                    if sizer
                    else prob * conf
                )
                size = base_size * scale
                payload = {
                    "Timestamp": data.get("Timestamp", ""),
                    "Symbol": symbol,
                    "prob": prob,
                    "confidence": conf,
                    "var": var,
                    "es": es,
                    "size": size,
                    "interval": (float(lower[0]), float(upper[0])),
                    "interval_width": width,
                }
                try:
                    realised = None
                    try:
                        realised = data.get("future_return")  # type: ignore[union-attr]
                    except Exception:
                        pass
                    if realised is None:
                        try:
                            realised = getattr(sig, "future_return")  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    if realised is None:
                        try:
                            realised = payload.get("future_return")  # type: ignore[union-attr]
                        except Exception:
                            pass
                    log_quantile_forecast(payload.get("Timestamp", ""), symbol, 0.05, var, None if realised is None else float(realised))
                except Exception:
                    pass
                if entry_scorer is not None:
                    depth = float(data.get("depth_imbalance", 0.0))
                    vol = float(data.get("volatility_30", 0.0))
                    embed = data.get("regime_embed", [])
                    score = entry_scorer.score(depth, vol, embed)
                    payload["expected_value"] = score
                    realised = float(data.get("future_return", 0.0))
                    try:
                        log_entry_value(payload.get("Timestamp", ""), symbol, score, realised)
                    except Exception:
                        pass
                    if score <= 0:
                        continue
                if not _passes_meta(meta_clf, prob, conf):
                    continue
                _EVENT_STORE.record("prediction", payload)
                yield payload
            else:
                sig = signals_pb2.Signal()
                sig.ParseFromString(msg.value)
                prob = float(sig.probability)
                symbol = sig.symbol
                conf = float(getattr(sig, "confidence", 1.0))
                var = getattr(sig, "var", None)
                es = getattr(sig, "es", None)
                if var is None or es is None:
                    q_var, q_es = _predict_var_es(prob, conf)
                    if var is None:
                        var = q_var
                    if es is None:
                        es = q_es
                if var is not None:
                    var = float(var)
                if es is not None:
                    es = float(es)
                lower, upper = conformal.predict_interval([prob], _CONFORMAL_Q)
                width = float(upper[0] - lower[0])
                if width > _MAX_INTERVAL_WIDTH:
                    if _DROP_WIDE_INTERVALS:
                        continue
                    scale = _MAX_INTERVAL_WIDTH / width if _MAX_INTERVAL_WIDTH != float("inf") else 0.0
                else:
                    scale = 1.0
                base_size = (
                    sizer.size(prob, symbol, var=var, es=es, confidence=conf)
                    if sizer
                    else prob * conf
                )
                size = base_size * scale
                payload = {
                    "Timestamp": sig.timestamp,
                    "Symbol": symbol,
                    "prob": prob,
                    "confidence": conf,
                    "var": var,
                    "es": es,
                    "size": size,
                    "interval": (float(lower[0]), float(upper[0])),
                    "interval_width": width,
                }
                try:
                    realised = None
                    try:
                        realised = data.get("future_return")  # type: ignore[union-attr]
                    except Exception:
                        pass
                    if realised is None:
                        try:
                            realised = getattr(sig, "future_return")  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    if realised is None:
                        try:
                            realised = payload.get("future_return")  # type: ignore[union-attr]
                        except Exception:
                            pass
                    log_quantile_forecast(payload.get("Timestamp", ""), symbol, 0.05, var, None if realised is None else float(realised))
                except Exception:
                    pass
                if entry_scorer is not None:
                    depth = float(getattr(sig, "depth_imbalance", 0.0))
                    vol = float(getattr(sig, "volatility_30", 0.0))
                    embed = getattr(sig, "regime_embed", [])
                    score = entry_scorer.score(depth, vol, embed)
                    payload["expected_value"] = score
                    realised = float(getattr(sig, "future_return", 0.0))
                    try:
                        log_entry_value(payload.get("Timestamp", ""), symbol, score, realised)
                    except Exception:
                        pass
                    if score <= 0:
                        continue
                if not _passes_meta(meta_clf, prob, conf):
                    continue
                _EVENT_STORE.record("prediction", payload)
                yield payload


class RedisSignalQueue:
    """Durable signal queue backed by Redis Streams."""

    def __init__(
        self,
        stream: str = "signals",
        url: str | None = None,
        start_id: str = "0-0",
    ) -> None:
        self.stream = stream
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.client = redis.Redis.from_url(self.url)
        self.last_id = start_id
        self.offset = 0

    def publish_dataframe(self, df: pd.DataFrame, fmt: str = "protobuf", retries: int = 3) -> None:
        fmt = fmt.lower()
        algos: list[str] = []
        sizes: list[float] = []
        for _, row in df.iterrows():
            feats = {
                "volatility": float(row.get("volatility_30", 0.0)),
                "trend_strength": float(row.get("trend_strength", 0.0)),
            }
            algo, action = _ROUTER.act(feats)
            algos.append(algo)
            sizes.append(float(action))
            if fmt == "json":
                payload_dict = {
                    "Timestamp": str(row["Timestamp"]),
                    "Symbol": str(row.get("Symbol", "")),
                    "prob": float(row["prob"]),
                    "confidence": float(row.get("confidence", 1.0)),
                }
                if "var" in row:
                    payload_dict["var"] = float(row["var"])
                if "es" in row:
                    payload_dict["es"] = float(row["es"])
                for field in [
                    "depth_imbalance",
                    "volatility_30",
                    "regime_embed",
                    "future_return",
                    "total_depth",
                ]:
                    if field in row:
                        payload_dict[field] = row[field]
                payload = json.dumps(payload_dict).encode()
            else:
                msg = signals_pb2.Signal(
                    timestamp=str(row["Timestamp"]),
                    symbol=str(row.get("Symbol", "")),
                    probability=str(row["prob"]),
                    confidence=float(row.get("confidence", 1.0)),
                )
                payload = msg.SerializeToString()
            for attempt in range(retries):
                try:
                    global _QUEUE_DEPTH
                    _QUEUE_DEPTH += 1
                    try:
                        record_metric("queue_depth", _QUEUE_DEPTH)
                    except Exception:
                        pass
                    self.client.xadd(self.stream, {"data": payload})
                    try:
                        record_metric("queue_offset", self.client.xlen(self.stream))
                    except Exception:
                        pass
                    break
                except redis.exceptions.RedisError:
                    if attempt + 1 == retries:
                        raise
                    time.sleep(1)
        df_log = df.copy()
        if "Timestamp" in df_log.columns:
            df_log = df_log.rename(columns={"Timestamp": "timestamp"})
        df_log["algorithm"] = algos
        df_log["position_size"] = sizes
        decision_logger.log(df_log)

    def iter_messages(
        self,
        fmt: str = "protobuf",
        sizer: PositionSizer | None = None,
        meta_clf: Any | None = None,
        entry_scorer: EntryValueScorer | None = None,
    ):
        fmt = fmt.lower()
        while True:
            resp = self.client.xread({self.stream: self.last_id}, count=1, block=1000)
            if not resp:
                continue
            _, messages = resp[0]
            for msg_id, fields in messages:
                data = fields[b"data"] if b"data" in fields else fields["data"]
                global _QUEUE_DEPTH
                _QUEUE_DEPTH = max(0, _QUEUE_DEPTH - 1)
                try:
                    record_metric("queue_depth", _QUEUE_DEPTH)
                except Exception:
                    pass
                self.last_id = msg_id
                try:
                    try:
                        record_metric("queue_offset", int(msg_id.split(b"-")[0]))
                    except Exception:
                        pass
                except Exception:
                    pass
                if fmt == "json":
                    payload = json.loads(data.decode())
                    symbol = payload.get("Symbol", "")
                    prob = float(payload.get("prob", 0.0))
                    conf = float(payload.get("confidence", 1.0))
                    var = payload.get("var")
                    es = payload.get("es")
                    lower, upper = conformal.predict_interval([prob], _CONFORMAL_Q)
                    width = float(upper[0] - lower[0])
                    if width > _MAX_INTERVAL_WIDTH:
                        if _DROP_WIDE_INTERVALS:
                            continue
                        scale = _MAX_INTERVAL_WIDTH / width if _MAX_INTERVAL_WIDTH != float("inf") else 0.0
                    else:
                        scale = 1.0
                    base_size = (
                        sizer.size(prob, symbol, var=var, es=es, confidence=conf)
                        if sizer
                        else prob * conf
                    )
                    size = base_size * scale
                    out = {
                        "Timestamp": payload.get("Timestamp", ""),
                        "Symbol": symbol,
                        "prob": prob,
                        "confidence": conf,
                        "var": var,
                        "es": es,
                        "size": size,
                        "interval": (float(lower[0]), float(upper[0])),
                        "interval_width": width,
                    }
                    try:
                        realised = None
                        try:
                            realised = data.get("future_return")  # type: ignore[union-attr]
                        except Exception:
                            pass
                        if realised is None:
                            try:
                                realised = getattr(sig, "future_return")  # type: ignore[attr-defined]
                            except Exception:
                                pass
                        if realised is None:
                            try:
                                realised = payload.get("future_return")  # type: ignore[union-attr]
                            except Exception:
                                pass
                        log_quantile_forecast(payload.get("Timestamp", ""), symbol, 0.05, var, None if realised is None else float(realised))
                    except Exception:
                        pass
                    if entry_scorer is not None:
                        depth = float(payload.get("depth_imbalance", 0.0))
                        vol = float(payload.get("volatility_30", 0.0))
                        embed = payload.get("regime_embed", [])
                        score = entry_scorer.score(depth, vol, embed)
                        out["expected_value"] = score
                        realised = float(payload.get("future_return", 0.0))
                        try:
                            log_entry_value(out.get("Timestamp", ""), symbol, score, realised)
                        except Exception:
                            pass
                        if score <= 0:
                            continue
                    if not _passes_meta(meta_clf, prob, conf):
                        continue
                    _EVENT_STORE.record("prediction", out)
                    yield out
                else:
                    sig = signals_pb2.Signal()
                    sig.ParseFromString(data)
                    prob = float(sig.probability)
                    symbol = sig.symbol
                    conf = float(getattr(sig, "confidence", 1.0))
                    var = getattr(sig, "var", None)
                    es = getattr(sig, "es", None)
                    if var is None or es is None:
                        q_var, q_es = _predict_var_es(prob, conf)
                        if var is None:
                            var = q_var
                        if es is None:
                            es = q_es
                    if var is not None:
                        var = float(var)
                    if es is not None:
                        es = float(es)
                    lower, upper = conformal.predict_interval([prob], _CONFORMAL_Q)
                    width = float(upper[0] - lower[0])
                    if width > _MAX_INTERVAL_WIDTH:
                        if _DROP_WIDE_INTERVALS:
                            continue
                        scale = _MAX_INTERVAL_WIDTH / width if _MAX_INTERVAL_WIDTH != float("inf") else 0.0
                    else:
                        scale = 1.0
                    base_size = (
                        sizer.size(prob, symbol, var=var, es=es, confidence=conf)
                        if sizer
                        else prob * conf
                    )
                    size = base_size * scale
                    out = {
                        "Timestamp": sig.timestamp,
                        "Symbol": symbol,
                        "prob": prob,
                        "confidence": conf,
                        "var": var,
                        "es": es,
                        "size": size,
                        "interval": (float(lower[0]), float(upper[0])),
                        "interval_width": width,
                    }
                    try:
                        realised = None
                        try:
                            realised = data.get("future_return")  # type: ignore[union-attr]
                        except Exception:
                            pass
                        if realised is None:
                            try:
                                realised = getattr(sig, "future_return")  # type: ignore[attr-defined]
                            except Exception:
                                pass
                        if realised is None:
                            try:
                                realised = payload.get("future_return")  # type: ignore[union-attr]
                            except Exception:
                                pass
                        log_quantile_forecast(payload.get("Timestamp", ""), symbol, 0.05, var, None if realised is None else float(realised))
                    except Exception:
                        pass
                    if entry_scorer is not None:
                        depth = float(getattr(sig, "depth_imbalance", 0.0))
                        vol = float(getattr(sig, "volatility_30", 0.0))
                        embed = getattr(sig, "regime_embed", [])
                        score = entry_scorer.score(depth, vol, embed)
                        out["expected_value"] = score
                        realised = float(getattr(sig, "future_return", 0.0))
                        try:
                            log_entry_value(out.get("Timestamp", ""), symbol, score, realised)
                        except Exception:
                            pass
                        if score <= 0:
                            continue
                    if not _passes_meta(meta_clf, prob, conf):
                        continue
                    _EVENT_STORE.record("prediction", out)
                    yield out


def get_signal_backend(cfg: dict | None):
    """Return a durable signal queue backend instance based on configuration."""
    cfg = cfg or {}
    backend = cfg.get("signal_backend", "zmq").lower()
    if backend == "kafka":
        return KafkaSignalQueue(
            topic=cfg.get("signal_topic", "signals"),
            bootstrap_servers=cfg.get("kafka_servers"),
            group_id=cfg.get("signal_group", "signal-queue"),
        )
    if backend == "redis":
        return RedisSignalQueue(
            stream=cfg.get("signal_stream", "signals"),
            url=cfg.get("redis_url"),
        )
    return None
