import os
import json
import time
import pandas as pd
import zmq
import zmq.asyncio
from typing import AsyncGenerator
from contextlib import contextmanager, asynccontextmanager

from kafka import KafkaProducer, KafkaConsumer, TopicPartition, errors as kafka_errors
import redis

from metrics import QUEUE_DEPTH, OFFSET_GAUGE

import asyncio
from proto import signals_pb2

_CTX = zmq.Context.instance()
_ASYNC_CTX = zmq.asyncio.Context.instance()


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
            }
            sock.send_json(payload)
        else:
            msg = signals_pb2.Signal(
                timestamp=str(row["Timestamp"]),
                symbol=str(row.get("Symbol", "")),
                probability=str(row["prob"]),
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
            }
            await sock.send_json(payload)
        else:
            msg = signals_pb2.Signal(
                timestamp=str(row["Timestamp"]),
                symbol=str(row.get("Symbol", "")),
                probability=str(row["prob"]),
            )
            await sock.send(msg.SerializeToString())


async def iter_messages(
    sock: zmq.asyncio.Socket, fmt: str = "protobuf"
) -> AsyncGenerator[dict, None]:
    """Yield decoded messages from a subscriber socket as they arrive."""
    fmt = fmt.lower()
    while True:
        if fmt == "json":
            data = await sock.recv_json()
            QUEUE_DEPTH.dec()
            yield {
                "Timestamp": data.get("Timestamp", ""),
                "Symbol": data.get("Symbol", ""),
                "prob": float(data.get("prob", 0.0)),
            }
        else:
            raw = await sock.recv()
            sig = signals_pb2.Signal()
            sig.ParseFromString(raw)
            QUEUE_DEPTH.dec()
            yield {
                "Timestamp": sig.timestamp,
                "Symbol": sig.symbol,
                "prob": float(sig.probability),
            }


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
                payload = json.dumps(
                    {
                        "Timestamp": str(row["Timestamp"]),
                        "Symbol": str(row.get("Symbol", "")),
                        "prob": float(row["prob"]),
                    }
                ).encode()
            else:
                msg = signals_pb2.Signal(
                    timestamp=str(row["Timestamp"]),
                    symbol=str(row.get("Symbol", "")),
                    probability=str(row["prob"]),
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

    def iter_messages(self, fmt: str = "protobuf"):
        fmt = fmt.lower()
        for msg in self.consumer:
            QUEUE_DEPTH.dec()
            self.offset = msg.offset
            OFFSET_GAUGE.set(self.offset)
            if fmt == "json":
                data = json.loads(msg.value.decode())
                yield {
                    "Timestamp": data.get("Timestamp", ""),
                    "Symbol": data.get("Symbol", ""),
                    "prob": float(data.get("prob", 0.0)),
                }
            else:
                sig = signals_pb2.Signal()
                sig.ParseFromString(msg.value)
                yield {
                    "Timestamp": sig.timestamp,
                    "Symbol": sig.symbol,
                    "prob": float(sig.probability),
                }


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
                payload = json.dumps(
                    {
                        "Timestamp": str(row["Timestamp"]),
                        "Symbol": str(row.get("Symbol", "")),
                        "prob": float(row["prob"]),
                    }
                ).encode()
            else:
                msg = signals_pb2.Signal(
                    timestamp=str(row["Timestamp"]),
                    symbol=str(row.get("Symbol", "")),
                    probability=str(row["prob"]),
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

    def iter_messages(self, fmt: str = "protobuf"):
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
                    yield {
                        "Timestamp": payload.get("Timestamp", ""),
                        "Symbol": payload.get("Symbol", ""),
                        "prob": float(payload.get("prob", 0.0)),
                    }
                else:
                    sig = signals_pb2.Signal()
                    sig.ParseFromString(data)
                    yield {
                        "Timestamp": sig.timestamp,
                        "Symbol": sig.symbol,
                        "prob": float(sig.probability),
                    }


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
