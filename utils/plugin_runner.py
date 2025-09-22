"""Utilities for running plugins in isolated subprocesses.

Plugins may be third-party code and could misbehave (e.g. hang or crash).
This module provides helpers that execute plugin callables in a separate
process with a timeout so that a hung plugin cannot block the main
application.
"""

from __future__ import annotations

import importlib
import logging
import multiprocessing as mp
import queue
import sys
from pathlib import Path
from typing import Any, Callable

try:  # pragma: no cover - ``resource`` is Unix only
    import resource
except Exception:  # pragma: no cover - Windows or limited platforms
    resource = None  # type: ignore

try:  # pragma: no cover - optional dependency guard
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers.polling import PollingObserver
except Exception:  # pragma: no cover - watchdog not installed
    FileSystemEventHandler = object  # type: ignore
    PollingObserver = None  # type: ignore
from mt5.metrics import PLUGIN_RELOADS


logger = logging.getLogger(__name__)


class PluginTimeoutError(RuntimeError):
    """Raised when a plugin does not finish execution within the timeout."""


def _plugin_entry(
    q: mp.Queue,
    func: Callable[..., Any],
    mem_limit_mb: float | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Execute ``func`` with optional memory limits and return via ``q``."""
    if mem_limit_mb and resource is not None:  # pragma: no cover - platform dependent
        try:
            limit = int(mem_limit_mb * 1024 * 1024)
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except Exception:
            # Memory limits are best-effort; failure to set should not crash
            pass
    try:
        q.put(("result", func(*args, **kwargs)))
    except Exception as exc:  # pragma: no cover - plugin errors are forwarded
        q.put(("error", exc))


def run_plugin(
    func: Callable[..., Any] | Any,
    *args: Any,
    timeout: float = 5.0,
    memory_limit_mb: float | None = None,
    **kwargs: Any,
) -> Any:
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
    proc = mp.Process(target=_plugin_entry, args=(q, target, memory_limit_mb, args, kwargs))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise PluginTimeoutError(f"Plugin '{getattr(func, 'name', target.__name__)}' timed out")
    if proc.exitcode not in (0, None):
        logger.error(
            "Sandboxed plugin %s exited with code %s",
            getattr(func, "name", getattr(target, "__name__", "unknown")),
            proc.exitcode,
        )
    try:
        status, value = q.get_nowait()
    except queue.Empty:
        logger.error("Plugin %s produced no result", getattr(func, "name", target.__name__))
        return None
    if status == "error":
        logger.error(
            "Plugin %s raised %s", getattr(func, "name", target.__name__), value
        )
        raise value
    return value


# --- live reloading -------------------------------------------------------


def _verify_import(module: str, q: mp.Queue) -> None:
    """Import ``module`` in a child process and report success or failure."""
    try:
        importlib.reload(importlib.import_module(module))
        q.put(("ok", None))
    except Exception as exc:  # pragma: no cover - propagated to parent
        q.put(("error", repr(exc)))


def _swap_in_module(module: str) -> None:
    """Reload ``module`` in the main process after validation."""
    if module in sys.modules:
        importlib.reload(sys.modules[module])
    else:
        importlib.import_module(module)


def _reload_plugin(module: str) -> None:
    """Attempt to hot-reload ``module`` after verifying it imports cleanly."""
    if PollingObserver is None:  # pragma: no cover - watchdog missing
        return
    q: mp.Queue = mp.Queue()
    proc = mp.Process(target=_verify_import, args=(module, q))
    proc.start()
    proc.join(5.0)
    if proc.exitcode not in (0, None):
        logger.warning("Plugin reload process for %s exited with %s", module, proc.exitcode)
        return
    try:
        status, payload = q.get_nowait()
    except queue.Empty:  # pragma: no cover - defensive
        status, payload = "error", "no status"
    if status != "ok":
        logger.warning("Failed to reload plugin %s: %s", module, payload)
        return
    try:
        _swap_in_module(module)
        PLUGIN_RELOADS.inc()
        logger.info("Reloaded plugin %s", module)
    except Exception:  # pragma: no cover - defensive
        logger.warning("Failed to swap in plugin %s", module, exc_info=True)


class _ReloadHandler(FileSystemEventHandler):
    """Watchdog handler that hot-reloads modified plugin modules."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _to_module(self, path: str) -> str | None:
        p = Path(path)
        if p.suffix != ".py":
            return None
        try:
            rel = p.relative_to(self.root)
        except ValueError:
            return None
        return f"plugins.{rel.stem}"

    def on_modified(self, event) -> None:  # type: ignore[override]
        if getattr(event, "is_directory", False):
            return
        mod = self._to_module(event.src_path)
        if mod:
            _reload_plugin(mod)

    on_created = on_modified


def _start_watcher() -> None:
    if PollingObserver is None:  # pragma: no cover - watchdog missing
        return
    root = Path(__file__).resolve().parents[1] / "plugins"
    observer = PollingObserver()
    handler = _ReloadHandler(root)
    observer.schedule(handler, str(root), recursive=False)
    observer.daemon = True
    observer.start()


_start_watcher()
