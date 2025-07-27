import os
import pandas as pd
import zmq
import zmq.asyncio
from typing import AsyncGenerator

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


def publish_dataframe(sock: zmq.Socket, df: pd.DataFrame) -> None:
    """Publish rows of a dataframe as Protobuf messages."""
    for _, row in df.iterrows():
        msg = signals_pb2.Signal(
            timestamp=str(row["Timestamp"]),
            symbol=str(row.get("Symbol", "")),
            probability=str(row["prob"]),
        )
        sock.send(msg.SerializeToString())


async def publish_dataframe_async(sock: zmq.asyncio.Socket, df: pd.DataFrame) -> None:
    """Asynchronously publish rows of a dataframe as Protobuf messages."""
    for _, row in df.iterrows():
        msg = signals_pb2.Signal(
            timestamp=str(row["Timestamp"]),
            symbol=str(row.get("Symbol", "")),
            probability=str(row["prob"]),
        )
        await sock.send(msg.SerializeToString())


async def iter_messages(sock: zmq.asyncio.Socket) -> AsyncGenerator[dict, None]:
    """Yield decoded messages from a subscriber socket as they arrive."""
    while True:
        raw = await sock.recv()
        sig = signals_pb2.Signal()
        sig.ParseFromString(raw)
        yield {"Timestamp": sig.timestamp, "Symbol": sig.symbol, "prob": float(sig.probability)}
