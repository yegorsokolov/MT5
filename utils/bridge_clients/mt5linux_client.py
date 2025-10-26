# utils/bridge_clients/mt5linux_client.py
"""
Lightweight client for the optional MT5 Windows RPC bridge (RPyC "classic" server).

Notes
-----
- Compatible with RPyC 5.3.1+ where `rpyc.classic.connect()` does NOT accept `config=`.
- If protocol tweaks are needed, we merge them into `rpyc.core.protocol.DEFAULT_CONFIG`
  before connecting (safe no-op on older/newer versions).
- If you're using the direct MetaTrader5 Python package under Wine (recommended),
  you can ignore this client entirely.

Typical usage
-------------
    from utils.bridge_clients.mt5linux_client import connect_to_win_bridge, rpc_ping

    conn = connect_to_win_bridge("127.0.0.1", 8765)
    try:
        ok, rtt = rpc_ping(conn, tries=1)
        print("ping:", ok, "rtt:", rtt)
        # Example: call into remote Python
        remote_time = conn.modules.time.time()
        print("remote time:", remote_time)
    finally:
        conn.close()
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple


__all__ = [
    "connect_to_win_bridge",
    "rpc_ping",
    "WinBridge",
]


def _merge_protocol_config_if_any(config: Optional[Dict[str, Any]]) -> None:
    """
    Merge user-supplied protocol options into RPyC's DEFAULT_CONFIG (if available).

    RPyC 5.3.1 classic.connect() signature:
        connect(host, port=18812, ipv6=False, keepalive=False, timeout=None)
    It does not accept `config=`; the recommended way is to mutate DEFAULT_CONFIG.
    """
    if not config:
        return
    try:
        import rpyc  # noqa: F401
        # Be defensive: guard all attribute lookups in case the layout changes in future.
        core = getattr(rpyc, "core", None)
        if core is None:
            return
        protocol = getattr(core, "protocol", None)
        if protocol is None:
            return
        default_cfg = getattr(protocol, "DEFAULT_CONFIG", None)
        if isinstance(default_cfg, dict):
            default_cfg.update(config)
    except Exception:
        # Swallow config merge issues silently to avoid breaking callers.
        pass


def connect_to_win_bridge(
    host: str = "127.0.0.1",
    port: int = 8765,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Connect to the Windows-side RPyC classic server without using an unsupported `config=` kwarg.

    Parameters
    ----------
    host : str
        Bridge host (usually 127.0.0.1).
    port : int
        Bridge port.
    config : dict | None
        Optional RPyC protocol options. Will be merged into DEFAULT_CONFIG.

    Returns
    -------
    rpyc.Connection
        A classic connection object. Caller is responsible for .close().
    """
    # Apply protocol tweaks if any (safe no-op if not supported)
    _merge_protocol_config_if_any(config)

    import rpyc
    # Classic connect (no `config=` kwarg!)
    return rpyc.classic.connect(host, port=port)


def rpc_ping(conn, tries: int = 1) -> Tuple[bool, float]:
    """
    Minimal ping using the remote interpreter (classic mode has no built-in ping).

    We attempt to import time on the remote and measure a round-trip using a trivial call.

    Parameters
    ----------
    conn : rpyc.Connection
        An active classic connection.
    tries : int
        Number of attempts; success if any attempt works.

    Returns
    -------
    (ok, rtt_seconds) : (bool, float)
        ok=True if at least one attempt succeeded. rtt is the last attempt's RTT.
    """
    ok = False
    last_rtt = 0.0
    for _ in range(max(1, tries)):
        t0 = time.perf_counter()
        try:
            # Touch a trivial attribute to exercise the channel.
            _ = conn.modules.time.time()
            ok = True
        except Exception:
            ok = False
        finally:
            last_rtt = time.perf_counter() - t0
        if ok:
            break
    return ok, last_rtt


@contextmanager
def WinBridge(
    host: str = "127.0.0.1",
    port: int = 8765,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Context manager wrapper around `connect_to_win_bridge`.

    Example
    -------
        with WinBridge("127.0.0.1", 8765) as conn:
            ok, rtt = rpc_ping(conn)
            print(ok, rtt)
    """
    conn = connect_to_win_bridge(host=host, port=port, config=config)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass
