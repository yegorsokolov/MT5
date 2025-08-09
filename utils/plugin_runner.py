"""Utilities for running plugins in isolated subprocesses.

Plugins may be third-party code and could misbehave (e.g. hang or crash).
This module provides helpers that execute plugin callables in a separate
process with a timeout so that a hung plugin cannot block the main
application.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
from typing import Any, Callable


class PluginTimeoutError(RuntimeError):
    """Raised when a plugin does not finish execution within the timeout."""


def _plugin_entry(q: mp.Queue, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Execute ``func`` and put the result or exception into ``q``."""
    try:
        q.put(("result", func(*args, **kwargs)))
    except Exception as exc:  # pragma: no cover - plugin errors are forwarded
        q.put(("error", exc))


def run_plugin(func: Callable[..., Any] | Any, *args: Any, timeout: float = 5.0, **kwargs: Any) -> Any:
    """Run ``func`` in a separate process with ``timeout`` seconds.

    Parameters
    ----------
    func:
        The plugin callable or an object implementing ``__call__``.
    timeout:
        Maximum number of seconds to wait for the plugin to finish.

    Returns
    -------
    Any
        The return value of ``func``.

    Raises
    ------
    PluginTimeoutError
        If the plugin does not finish within ``timeout`` seconds.
    Exception
        Any exception raised inside the plugin is re-raised in the caller.
    """

    target = getattr(func, "plugin", func)  # Support PluginSpec objects
    q: mp.Queue = mp.Queue()
    proc = mp.Process(target=_plugin_entry, args=(q, target, *args), kwargs=kwargs)
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise PluginTimeoutError(f"Plugin '{getattr(func, 'name', target.__name__)}' timed out")
    try:
        status, value = q.get_nowait()
    except queue.Empty:
        return None
    if status == "error":
        raise value
    return value
