#!/usr/bin/env python3
"""Minimal RPyC bridge server exposing the MetaTrader5 Python API.

The historical deployment relied on the third-party ``mt5linux`` package to
spawn an RPyC server inside the Wine-hosted Python interpreter. When Wine
struggles with the packaged launcher we can fall back to this tiny helper. The
server loads ``MetaTrader5`` (or the ``mt5`` compatibility shim) and then starts
an RPyC ``SlaveService`` so Linux clients can proxy the module transparently.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from typing import Any

import rpyc
from rpyc.utils.classic import DEFAULT_SERVER_PORT, SlaveService
from rpyc.utils.server import ThreadedServer

_LOG = logging.getLogger("mt5_rpyc_server")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch an RPyC bridge for MetaTrader5")
    parser.add_argument("--host", default="127.0.0.1", help="Hostname for the RPyC listener")
    parser.add_argument(
        "--port",
        default=DEFAULT_SERVER_PORT,
        type=int,
        help="TCP port for the RPyC listener (default: %(default)s)",
    )
    parser.add_argument(
        "--module",
        default=None,
        help="Optional preferred module name to import (fallbacks: MetaTrader5, mt5)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices={"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"},
        help="Logging verbosity for the helper",
    )
    parser.add_argument(
        "--timeout",
        default=30.0,
        type=float,
        help="Sync request timeout passed to the RPyC server (seconds)",
    )
    return parser.parse_args()


def _load_bridge_module(preferred: str | None) -> tuple[Any, str]:
    """Attempt to import the requested MetaTrader5 bridge module."""

    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["MetaTrader5", "mt5"])

    errors: list[str] = []
    for name in candidates:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - depends on Windows setup
            errors.append(f"{name}: {exc}")
            continue

        # Some drop-in replacements (such as the pip 'mt5' package) expose the
        # API under a different module name. Ensure the classic import path is
        # available for bridge clients.
        if name != "MetaTrader5":
            sys.modules.setdefault("MetaTrader5", module)
        return module, name

    raise ImportError("; ".join(errors))


def main() -> None:
    args = _parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    _LOG.info("Starting MetaTrader5 RPyC bridge on %s:%s", args.host, args.port)

    try:
        module, module_name = _load_bridge_module(args.module)
    except Exception as exc:  # pragma: no cover - depends on runtime
        _LOG.error("Failed to import MetaTrader5 bridge module: %s", exc)
        raise SystemExit(1) from exc

    module_path = getattr(module, "__file__", "<unknown>")
    _LOG.info("Loaded module '%s' from %s", module_name, module_path)

    initializer = getattr(module, "initialize", None)
    if callable(initializer):
        try:
            initializer()
        except Exception as exc:  # pragma: no cover - depends on terminal state
            _LOG.warning("Module initialize() failed: %s", exc)

    config = {
        "allow_all_attrs": True,
        "allow_pickle": True,
        "allow_public_attrs": True,
        "sync_request_timeout": max(1.0, float(args.timeout)),
    }

    server = ThreadedServer(
        SlaveService,
        hostname=args.host,
        port=args.port,
        reuse_addr=True,
        protocol_config=config,
    )

    try:
        server.start()
    except KeyboardInterrupt:  # pragma: no cover - manual shutdown
        _LOG.info("Received interrupt; shutting down bridge server")


if __name__ == "__main__":  # pragma: no branch
    main()
