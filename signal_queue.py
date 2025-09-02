"""Lightweight in-memory signal queue without external dependencies."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any

import pandas as pd


class InMemoryQueue:
    """A minimal async queue for passing signal dataframes."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[list[dict[str, Any]]] = asyncio.Queue()

    def publish_dataframe(self, df: pd.DataFrame, fmt: str = "json") -> None:
        self._queue.put_nowait(df.to_dict(orient="records"))

    async def publish_dataframe_async(self, df: pd.DataFrame, fmt: str = "json") -> None:
        self.publish_dataframe(df, fmt)

    async def iter_messages(self) -> AsyncGenerator[dict[str, Any], None]:
        while True:
            batch = await self._queue.get()
            for msg in batch:
                yield msg


def get_signal_backend(cfg: dict | None = None) -> InMemoryQueue:
    """Return a simple in-memory backend for signals."""
    return InMemoryQueue()


def publish_dataframe(sock: InMemoryQueue, df: pd.DataFrame, fmt: str = "json") -> None:
    sock.publish_dataframe(df, fmt)


async def publish_dataframe_async(sock: InMemoryQueue, df: pd.DataFrame, fmt: str = "json") -> None:
    await sock.publish_dataframe_async(df, fmt)


async def iter_messages(sock: InMemoryQueue, fmt: str = "json", sizer=None) -> AsyncGenerator[dict, None]:
    async for msg in sock.iter_messages():
        yield msg
