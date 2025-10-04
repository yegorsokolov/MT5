"""Client wrapper for the mt5linux RPyC bridge."""

from __future__ import annotations

import os
import threading
from types import ModuleType
from typing import Any

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 18812  # rpyc.utils.classic.DEFAULT_SERVER_PORT
_DEFAULT_TIMEOUT = 30.0

_LOCK = threading.Lock()
_CONN: Any | None = None
_META_MODULE: ModuleType | None = None
_BRIDGE_INFO: dict[str, Any] = {}

MetaTrader5: ModuleType | None = None


class _MetaTraderModule(ModuleType):
    """Proxy object that exposes the remote MetaTrader5 module API."""

    def __init__(self, *, conn: Any, remote_module: Any, host: str, port: int) -> None:
        super().__init__("MetaTrader5")
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_remote", remote_module)
        # Provide a synthetic path for debugging / describe_backend output.
        object.__setattr__(self, "__file__", f"rpyc://{host}:{port}/MetaTrader5")

    @property
    def _remote(self) -> Any:  # pragma: no cover - property helper for mypy
        return object.__getattribute__(self, "_remote")

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_remote"), name)

    def __dir__(self) -> list[str]:  # pragma: no cover - exercised indirectly
        return sorted(set(super().__dir__()) | set(dir(self._remote)))


def _read_host() -> tuple[str, str]:
    for key in ("MT5LINUX_HOST", "PYMT5LINUX_HOST", "MT5_BRIDGE_HOST"):
        value = os.getenv(key)
        if value:
            return value.strip() or _DEFAULT_HOST, key
    return _DEFAULT_HOST, "default"


def _read_port() -> tuple[int, str]:
    for key in ("MT5LINUX_PORT", "PYMT5LINUX_PORT", "MT5_BRIDGE_PORT"):
        value = os.getenv(key)
        if value:
            try:
                return int(value.strip()), key
            except ValueError as exc:
                raise RuntimeError(f"Invalid integer for {key}: {value}") from exc
    return _DEFAULT_PORT, "default"


def _read_timeout() -> float:
    value = os.getenv("MT5LINUX_TIMEOUT", os.getenv("MT5LINUX_TIMEOUT_SECONDS"))
    if value:
        try:
            return float(value.strip())
        except ValueError as exc:  # pragma: no cover - configuration error
            raise RuntimeError(f"Invalid timeout value: {value}") from exc
    return _DEFAULT_TIMEOUT


def _connect(host: str, port: int, timeout: float) -> Any:
    try:
        import rpyc
    except Exception as exc:  # pragma: no cover - optional dependency at runtime
        raise RuntimeError("rpyc package is required for the mt5linux client") from exc

    config = {
        "sync_request_timeout": timeout,
        "connection_timeout": timeout,
        "allow_public_attrs": True,
    }

    try:
        return rpyc.classic.connect(host, port=port, config=config)
    except OSError as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError(f"Unable to connect to mt5linux server at {host}:{port}: {exc}") from exc


def _ensure_connected(*, force: bool = False) -> ModuleType:
    global _CONN, _META_MODULE, MetaTrader5, _BRIDGE_INFO
    with _LOCK:
        if _META_MODULE is not None and not force:
            return _META_MODULE

        host, host_source = _read_host()
        port, port_source = _read_port()
        timeout = _read_timeout()

        if _CONN is not None:
            try:
                _CONN.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            _CONN = None

        conn = _connect(host, port, timeout)
        try:
            conn.execute("import MetaTrader5 as mt5")
            remote_module = conn.modules.MetaTrader5
        except Exception:
            conn.close()
            raise

        module = _MetaTraderModule(
            conn=conn,
            remote_module=remote_module,
            host=host,
            port=port,
        )

        _CONN = conn
        _META_MODULE = module
        MetaTrader5 = module
        _BRIDGE_INFO = {
            "host": host,
            "host_source": host_source,
            "port": port,
            "port_source": port_source,
            "timeout": timeout,
        }
        return module


def bridge_info() -> dict[str, Any]:
    """Return metadata about the active mt5linux connection."""

    with _LOCK:
        return dict(_BRIDGE_INFO)


def initialize(*, force: bool = False) -> None:
    """Initialise the mt5linux bridge by establishing the RPyC connection."""

    _ensure_connected(force=force)


def load_mt5(*, force: bool = False) -> ModuleType:
    """Return the MetaTrader5 module proxy from the mt5linux bridge."""

    return _ensure_connected(force=force)


def close() -> None:
    """Close the underlying RPyC connection."""  # pragma: no cover - exercised in integration

    global _CONN, _META_MODULE, MetaTrader5
    with _LOCK:
        if _CONN is not None:
            try:
                _CONN.close()
            except Exception:
                pass
        _CONN = None
        _META_MODULE = None
        MetaTrader5 = None
        _BRIDGE_INFO.clear()


__all__ = [
    "MetaTrader5",
    "bridge_info",
    "close",
    "initialize",
    "load_mt5",
]
