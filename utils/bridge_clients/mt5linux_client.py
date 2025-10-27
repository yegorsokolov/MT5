# utils/bridge_clients/mt5linux_client.py
from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

__all__ = ["connect_to_win_bridge", "rpc_ping", "WinBridge"]

def _merge_protocol_config_if_any(config: Optional[Dict[str, Any]]) -> None:
    if not config:
        return
    try:
        import rpyc  # noqa
        core = getattr(rpyc, "core", None)
        protocol = getattr(core, "protocol", None) if core else None
        default_cfg = getattr(protocol, "DEFAULT_CONFIG", None) if protocol else None
        if isinstance(default_cfg, dict):
            default_cfg.update(config)
    except Exception:
        pass

def connect_to_win_bridge(host: str = "127.0.0.1", port: int = 8765, config: Optional[Dict[str, Any]] = None):
    _merge_protocol_config_if_any(config)
    import rpyc
    return rpyc.classic.connect(host, port=port)

def rpc_ping(conn, tries: int = 1) -> Tuple[bool, float]:
    ok, last_rtt = False, 0.0
    for _ in range(max(1, tries)):
        t0 = time.perf_counter()
        try:
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
def WinBridge(host: str = "127.0.0.1", port: int = 8765, config: Optional[Dict[str, Any]] = None):
    conn = connect_to_win_bridge(host=host, port=port, config=config)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass
