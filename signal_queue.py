"""Signal queue built on the shared :mod:`services.message_bus`."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict

import logging
import pandas as pd

from services.message_bus import Topics, get_message_bus, MessageBus

from analysis import pipeline_anomaly

logger = logging.getLogger(__name__)

# ``_ROUTER`` is retained for compatibility with modules that import it.  The
# message bus supersedes the old ZeroMQ based router so it is simply ``None``
# here.
_ROUTER = None


def get_signal_backend(cfg: Dict[str, Any] | None = None) -> MessageBus:
    """Return a :class:`MessageBus` instance for publishing signals."""

    backend = (cfg or {}).get("signal_backend") if cfg else None
    return get_message_bus(backend)


# ---------------------------------------------------------------------------


def _wrap_ci(row: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap prediction rows with credible intervals if present."""

    if {"pred", "ci_lower", "ci_upper"}.issubset(row):
        row = row.copy()
        row["prediction"] = {
            "mean": row.pop("pred"),
            "lower": row.pop("ci_lower"),
            "upper": row.pop("ci_upper"),
        }
    return row


def _is_empty(df: pd.DataFrame) -> bool:
    return getattr(df, "empty", len(df) == 0)  # type: ignore[arg-type]


def _validate(df: pd.DataFrame) -> bool:
    return pipeline_anomaly.validate(df) if hasattr(df, "columns") else True


def publish_dataframe(bus: MessageBus, df: pd.DataFrame, fmt: str = "json") -> None:
    """Synchronously publish each row of ``df`` to the signals topic."""

    if _is_empty(df):
        return
    if not _validate(df):
        logger.warning("Pipeline anomaly detected; dropping batch")
        return
    rows = [_wrap_ci(r) for r in df.to_dict(orient="records")]

    async def _pub() -> None:
        for row in rows:
            await bus.publish(Topics.SIGNALS, row)

    asyncio.run(_pub())


async def publish_dataframe_async(
    bus: MessageBus, df: pd.DataFrame, fmt: str = "json"
) -> None:
    """Asynchronously publish each row of ``df`` to the signals topic."""

    if _is_empty(df):
        return
    if not _validate(df):
        logger.warning("Pipeline anomaly detected; dropping batch")
        return
    rows = [_wrap_ci(r) for r in df.to_dict(orient="records")]
    for row in rows:
        await bus.publish(Topics.SIGNALS, row)


async def iter_messages(
    bus: MessageBus, fmt: str = "json", sizer=None
) -> AsyncGenerator[Dict[str, Any], None]:
    """Yield messages from the signals topic."""

    async for msg in bus.subscribe(Topics.SIGNALS):
        yield msg
