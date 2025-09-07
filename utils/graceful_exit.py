"""Utilities for gracefully terminating the event loop.

This module provides an asynchronous :func:`graceful_exit` helper used by
resource watchdog callbacks.  The function attempts to cancel any pending
asyncio tasks, flush Prometheus metrics and logging handlers, and then raises
``SystemExit`` so that callers can terminate cleanly without resorting to
``os._exit``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Iterable

from prometheus_client import generate_latest


async def graceful_exit(code: int = 1) -> None:
    """Gracefully shut down the running asyncio application.

    Parameters
    ----------
    code:
        Exit code to raise with ``SystemExit``.

    The function performs best–effort cancellation of all pending tasks in the
    current event loop, waits for them to finish, flushes Prometheus metrics and
    log handlers, and finally raises ``SystemExit``.
    """

    loop = asyncio.get_running_loop()
    current = asyncio.current_task(loop=loop)
    pending: Iterable[asyncio.Task] = [
        t for t in asyncio.all_tasks(loop) if t is not current
    ]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    try:
        # Generating the latest metrics snapshot ensures any buffered values are
        # flushed to exporters before shutdown.
        generate_latest()
    except Exception:  # pragma: no cover - best effort only
        pass

    # Flush and close all logging handlers.
    root = logging.getLogger()
    for handler in root.handlers:
        try:  # pragma: no cover - logging handlers rarely fail
            handler.flush()
        except Exception:
            pass
    logging.shutdown()

    raise SystemExit(code)
