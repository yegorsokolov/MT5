import os
import json
import pandas as pd
import zmq
import zmq.asyncio
from typing import AsyncGenerator

from metrics import QUEUE_DEPTH

import asyncio
from proto import signals_pb2

_CTX = zmq.Context.instance()
_ASYNC_CTX = zmq.asyncio.Context.instance()


def get_publisher(bind_address: str | None = None) -> zmq.Socket:
    """Return a PUB socket bound to the given address."""
    addr = bind_address or os.getenv("SIGNAL_QUEUE_BIND", "tcp://*:5555")
    sock = _CTX.socket(zmq.PUB)
    sock.bind(addr)
    return sock


def get_subscriber(connect_address: str | None = None, topic: str = "") -> zmq.Socket:
    """Return a SUB socket connected to the given address."""
    addr = connect_address or os.getenv("SIGNAL_QUEUE_URL", "tcp://localhost:5555")
    sock = _CTX.socket(zmq.SUB)
    sock.connect(addr)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    return sock


def get_async_publisher(bind_address: str | None = None) -> zmq.asyncio.Socket:
    """Return an async PUB socket bound to the given address."""
    addr = bind_address or os.getenv("SIGNAL_QUEUE_BIND", "tcp://*:5555")
    sock = _ASYNC_CTX.socket(zmq.PUB)
    sock.bind(addr)
    return sock


def get_async_subscriber(connect_address: str | None = None, topic: str = "") -> zmq.asyncio.Socket:
    """Return an async SUB socket connected to the given address."""
    addr = connect_address or os.getenv("SIGNAL_QUEUE_URL", "tcp://localhost:5555")
    sock = _ASYNC_CTX.socket(zmq.SUB)
    sock.connect(addr)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    return sock


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
