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

from metrics import QUEUE_DEPTH, OFFSET_GAUGE
from risk.position_sizer import PositionSizer

import asyncio
from proto import signals_pb2
from event_store import EventStore
from pathlib import Path

_EVENT_STORE = EventStore(Path(os.getenv("EVENT_STORE_PATH", Path(__file__).resolve().parent / "data" / "events.db")))

_CTX = zmq.Context.instance()
_ASYNC_CTX = zmq.asyncio.Context.instance()


def _passes_meta(meta_clf: Any | None, prob: float, conf: float) -> bool:
    """Return True if a message passes the meta-classifier filter.

    The meta classifier is expected to implement ``predict`` with a
    signature compatible with scikit-learn estimators.  It receives a
    dataframe containing the message's probability and confidence.
    """

    if meta_clf is None:
        return True
    try:
        feats = pd.DataFrame([[prob, conf]], columns=["prob", "confidence"])
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


def publish_dataframe(sock: zmq.Socket, df: pd.DataFrame, fmt: str = "protobuf") -> None:
    """Publish rows of a dataframe as JSON or Protobuf messages."""
    fmt = fmt.lower()
    for _, row in df.iterrows():
        QUEUE_DEPTH.inc()
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
            sock.send_json(payload)
        else:
            msg = signals_pb2.Signal(
                timestamp=str(row["Timestamp"]),
                symbol=str(row.get("Symbol", "")),
                probability=str(row["prob"]),
                confidence=float(row.get("confidence", 1.0)),
            )
            sock.send(msg.SerializeToString())


async def publish_dataframe_async(
    sock: zmq.asyncio.Socket, df: pd.DataFrame, fmt: str = "protobuf"
) -> None:
    """Asynchronously publish rows of a dataframe as JSON or Protobuf messages."""
    fmt = fmt.lower()
    for _, row in df.iterrows():
        QUEUE_DEPTH.inc()
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
            await sock.send_json(payload)
        else:
            msg = signals_pb2.Signal(
                timestamp=str(row["Timestamp"]),
                symbol=str(row.get("Symbol", "")),
                probability=str(row["prob"]),
                confidence=float(row.get("confidence", 1.0)),
            )
            await sock.send(msg.SerializeToString())


async def iter_messages(
    sock: zmq.asyncio.Socket,
    fmt: str = "protobuf",
    sizer: PositionSizer | None = None,
    meta_clf: Any | None = None,
) -> AsyncGenerator[dict, None]:
    """Yield decoded messages from a subscriber socket as they arrive."""
    fmt = fmt.lower()
    while True:
        if fmt == "json":
            data = await sock.recv_json()
            QUEUE_DEPTH.dec()
            symbol = data.get("Symbol", "")
            prob = float(data.get("prob", 0.0))
            conf = float(data.get("confidence", 1.0))
            var = data.get("var")
            es = data.get("es")
            size = (
                sizer.size(prob, symbol, var=var, es=es, confidence=conf)
                if sizer
                else prob * conf
            )
            payload = {
                "Timestamp": data.get("Timestamp", ""),
                "Symbol": symbol,
                "prob": prob,
                "confidence": conf,
                "var": var,
                "es": es,
                "size": size,
            }
            if not _passes_meta(meta_clf, prob, conf):
                continue
            _EVENT_STORE.record("prediction", payload)
            yield payload
        else:
            raw = await sock.recv()
            sig = signals_pb2.Signal()
            sig.ParseFromString(raw)
            QUEUE_DEPTH.dec()
            prob = float(sig.probability)
            symbol = sig.symbol
            conf = float(getattr(sig, "confidence", 1.0))
            var = getattr(sig, "var", None)
            es = getattr(sig, "es", None)
            if var is not None:
                var = float(var)
            if es is not None:
                es = float(es)
            size = (
                sizer.size(prob, symbol, var=var, es=es, confidence=conf)
                if sizer
                else prob * conf
            )
            payload = {
                "Timestamp": sig.timestamp,
                "Symbol": symbol,
                "prob": prob,
                "confidence": conf,
                "var": var,
                "es": es,
                "size": size,
            }
            if not _passes_meta(meta_clf, prob, conf):
                continue
            _EVENT_STORE.record("prediction", payload)
            yield payload


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
        for _, row in df.iterrows():
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
                    QUEUE_DEPTH.inc()
                    future = self.producer.send(self.topic, payload)
                    meta = future.get(timeout=10)
                    self.offset = meta.offset
                    OFFSET_GAUGE.set(self.offset)
                    break
                except kafka_errors.KafkaError:
                    if attempt + 1 == retries:
                        raise
                    time.sleep(1)

    def iter_messages(
        self,
        fmt: str = "protobuf",
        sizer: PositionSizer | None = None,
        meta_clf: Any | None = None,
    ):
        fmt = fmt.lower()
        for msg in self.consumer:
            QUEUE_DEPTH.dec()
            self.offset = msg.offset
            OFFSET_GAUGE.set(self.offset)
            if fmt == "json":
                data = json.loads(msg.value.decode())
                symbol = data.get("Symbol", "")
                prob = float(data.get("prob", 0.0))
                conf = float(data.get("confidence", 1.0))
                var = data.get("var")
                es = data.get("es")
                size = (
                    sizer.size(prob, symbol, var=var, es=es, confidence=conf)
                    if sizer
                    else prob * conf
                )
                payload = {
                    "Timestamp": data.get("Timestamp", ""),
                    "Symbol": symbol,
                    "prob": prob,
                    "confidence": conf,
                    "var": var,
                    "es": es,
                    "size": size,
                }
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
                if var is not None:
                    var = float(var)
                if es is not None:
                    es = float(es)
                size = (
                    sizer.size(prob, symbol, var=var, es=es, confidence=conf)
                    if sizer
                    else prob * conf
                )
                payload = {
                    "Timestamp": sig.timestamp,
                    "Symbol": symbol,
                    "prob": prob,
                    "confidence": conf,
                    "var": var,
                    "es": es,
                    "size": size,
                }
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
        for _, row in df.iterrows():
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
                    QUEUE_DEPTH.inc()
                    self.client.xadd(self.stream, {"data": payload})
                    OFFSET_GAUGE.set(self.client.xlen(self.stream))
                    break
                except redis.exceptions.RedisError:
                    if attempt + 1 == retries:
                        raise
                    time.sleep(1)

    def iter_messages(
        self,
        fmt: str = "protobuf",
        sizer: PositionSizer | None = None,
        meta_clf: Any | None = None,
    ):
        fmt = fmt.lower()
        while True:
            resp = self.client.xread({self.stream: self.last_id}, count=1, block=1000)
            if not resp:
                continue
            _, messages = resp[0]
            for msg_id, fields in messages:
                data = fields[b"data"] if b"data" in fields else fields["data"]
                QUEUE_DEPTH.dec()
                self.last_id = msg_id
                try:
                    OFFSET_GAUGE.set(int(msg_id.split(b"-")[0]))
                except Exception:
                    pass
                if fmt == "json":
                    payload = json.loads(data.decode())
                    symbol = payload.get("Symbol", "")
                    prob = float(payload.get("prob", 0.0))
                    conf = float(payload.get("confidence", 1.0))
                    var = payload.get("var")
                    es = payload.get("es")
                    size = (
                        sizer.size(prob, symbol, var=var, es=es, confidence=conf)
                        if sizer
                        else prob * conf
                    )
                    out = {
                        "Timestamp": payload.get("Timestamp", ""),
                        "Symbol": symbol,
                        "prob": prob,
                        "confidence": conf,
                        "var": var,
                        "es": es,
                        "size": size,
                    }
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
                    if var is not None:
                        var = float(var)
                    if es is not None:
                        es = float(es)
                    size = (
                        sizer.size(prob, symbol, var=var, es=es, confidence=conf)
                        if sizer
                        else prob * conf
                    )
                    out = {
                        "Timestamp": sig.timestamp,
                        "Symbol": symbol,
                        "prob": prob,
                        "confidence": conf,
                        "var": var,
                        "es": es,
                        "size": size,
                    }
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
