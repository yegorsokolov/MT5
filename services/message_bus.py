"""Unified message bus with NATS, Kafka and in-memory backends."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Deque, Dict, Optional

__all__ = ["MessageBus", "get_message_bus", "Topics", "TOPIC_CONFIG"]


@dataclass(frozen=True)
class Topics:
    """Well known topic names used throughout the system."""

    TICKS: str = "ticks"
    FEATURES: str = "features"
    SIGNALS: str = "signals"


# Simple retention/backpressure configuration for each topic.  The in-memory
# backend enforces these limits while external brokers should be configured with
# equivalent policies (see ``docs/message_bus.md`` for deployment notes).
TOPIC_CONFIG: Dict[str, Dict[str, Any]] = {
    Topics.TICKS: {"retention": 60, "max_msgs": 10_000},  # 60s retention in memory
    Topics.FEATURES: {"retention": 600, "max_msgs": 5_000},  # 10m retention
    Topics.SIGNALS: {"retention": 3_600, "max_msgs": 1_000},  # 1h retention
}


class MessageBus:
    """Very small abstraction layer over different messaging backends."""

    def __init__(self, backend: str = "inmemory", **kwargs: Any) -> None:
        self.backend = backend
        self.kwargs = kwargs
        self._queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self._buffers: Dict[str, Deque[tuple[float, Any]]] = defaultdict(deque)
        self._nats = None
        self._kafka_producer = None
        self._kafka_consumers: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    async def _ensure_nats(self) -> None:  # pragma: no cover - requires network
        if self._nats is None:
            import nats

            url = self.kwargs.get("url") or os.getenv(
                "NATS_URL", "nats://localhost:4222"
            )
            self._nats = await nats.connect(url)

    # ------------------------------------------------------------------
    async def _ensure_kafka(self) -> None:  # pragma: no cover - requires network
        if self._kafka_producer is None:
            from aiokafka import AIOKafkaProducer

            servers = self.kwargs.get("bootstrap_servers") or os.getenv(
                "KAFKA_BOOTSTRAP", "localhost:9092"
            )
            self._kafka_producer = AIOKafkaProducer(bootstrap_servers=servers)
            await self._kafka_producer.start()

    # ------------------------------------------------------------------
    async def publish(self, topic: str, msg: Any) -> None:
        """Publish ``msg`` to ``topic``.

        ``msg`` can be ``bytes`` or any JSON serialisable object.  For the
        in-memory backend, basic retention/backpressure is applied according to
        :data:`TOPIC_CONFIG`.
        """

        if self.backend == "nats":  # pragma: no cover - network heavy
            await self._ensure_nats()
            data = (
                msg if isinstance(msg, (bytes, bytearray)) else json.dumps(msg).encode()
            )
            await self._nats.publish(topic, data)
            return

        if self.backend == "kafka":  # pragma: no cover - network heavy
            await self._ensure_kafka()
            data = (
                msg if isinstance(msg, (bytes, bytearray)) else json.dumps(msg).encode()
            )
            await self._kafka_producer.send_and_wait(topic, data)
            return

        # Default in-memory implementation
        cfg = TOPIC_CONFIG.get(topic, {"max_msgs": 1000})
        queue = self._queues[topic]
        buf = self._buffers[topic]

        # drop oldest item from queue if it exceeds max size
        if queue.qsize() >= cfg["max_msgs"]:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

        now = time.time()
        buf.append((now, msg))

        # enforce retention policy
        retention = cfg.get("retention")
        if retention is not None:
            cutoff = now - retention
            while buf and buf[0][0] < cutoff:
                buf.popleft()

        # enforce buffer size limit
        if len(buf) > cfg.get("max_msgs", 1000):
            buf.popleft()

        await queue.put(msg)

    # ------------------------------------------------------------------
    def get_history(self, topic: str) -> list[Any]:
        """Return buffered messages for ``topic`` in publish order."""

        return [m for _, m in self._buffers.get(topic, [])]

    # ------------------------------------------------------------------
    async def subscribe(self, topic: str) -> AsyncGenerator[Any, None]:
        """Yield messages published on ``topic``."""

        if self.backend == "nats":  # pragma: no cover - network heavy
            await self._ensure_nats()
            q: asyncio.Queue = asyncio.Queue()

            async def _cb(msg):
                await q.put(msg.data)

            await self._nats.subscribe(topic, cb=_cb)
            while True:
                data = await q.get()
                yield json.loads(data)

        elif self.backend == "kafka":  # pragma: no cover - network heavy
            from aiokafka import AIOKafkaConsumer

            if topic not in self._kafka_consumers:
                servers = self.kwargs.get("bootstrap_servers") or os.getenv(
                    "KAFKA_BOOTSTRAP", "localhost:9092"
                )
                consumer = AIOKafkaConsumer(topic, bootstrap_servers=servers)
                await consumer.start()
                self._kafka_consumers[topic] = consumer
            consumer = self._kafka_consumers[topic]
            try:
                async for msg in consumer:
                    yield json.loads(msg.value)
            finally:  # pragma: no cover - cleanup
                await consumer.stop()
        else:
            queue = self._queues[topic]
            while True:
                data = await queue.get()
                yield data

    # ------------------------------------------------------------------
    async def close(self) -> None:
        """Close backend connections and flush async logs."""

        try:
            from mt5.log_utils import shutdown_logging

            shutdown_logging()
        except Exception:
            pass

        if self.backend == "nats" and self._nats is not None:  # pragma: no cover
            await self._nats.close()
            self._nats = None
        if self.backend == "kafka" and self._kafka_producer is not None:  # pragma: no cover
            await self._kafka_producer.stop()
            self._kafka_producer = None


# ----------------------------------------------------------------------
_message_bus: Optional[MessageBus] = None


def get_message_bus(backend: str | None = None, **kwargs: Any) -> MessageBus:
    """Return a singleton :class:`MessageBus` instance."""

    global _message_bus
    if _message_bus is None:
        backend = backend or os.getenv("MESSAGE_BUS_BACKEND", "inmemory")
        _message_bus = MessageBus(backend=backend, **kwargs)
    return _message_bus
