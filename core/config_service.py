"""Configuration service exposing REST and gRPC APIs.

This module provides a small configuration service backed by a SQLite
store.  Configuration values are versioned so that components can detect
changes.  Both REST (FastAPI) and gRPC endpoints are provided for
retrieving and updating configuration values.  Clients may subscribe to
change notifications and reload settings without restarting.

The service implements a very small access control layer (ACL) based on
API keys.  All configuration changes are recorded to an audit log so that
operators can trace who changed what and when.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List

import grpc
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from google.protobuf import empty_pb2

from proto import config_pb2, config_pb2_grpc

# ---------------------------------------------------------------------------
# Storage backend
# ---------------------------------------------------------------------------


class SQLiteConfigStore:
    """Very small SQLite backed key/value store with versioning."""

    def __init__(self, db_path: str = "config.db") -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                version INTEGER NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT,
                old_value TEXT,
                new_value TEXT,
                version INTEGER,
                user TEXT,
                reason TEXT,
                ts TEXT
            )
            """
        )
        self.conn.commit()
        self.lock = asyncio.Lock()
        self.subscribers: List[asyncio.Queue[Dict[str, Any]]] = []

    async def get(self, key: str) -> Dict[str, Any] | None:
        async with self.lock:
            def _get() -> Dict[str, Any] | None:
                cur = self.conn.execute(
                    "SELECT value, version FROM config WHERE key=?", (key,)
                )
                row = cur.fetchone()
                if row:
                    return {"key": key, "value": row[0], "version": row[1]}
                return None

            return await asyncio.to_thread(_get)

    async def set(self, key: str, value: str, user: str, reason: str) -> Dict[str, Any]:
        async with self.lock:
            def _set() -> Dict[str, Any]:
                cur = self.conn.execute(
                    "SELECT value, version FROM config WHERE key=?", (key,)
                )
                row = cur.fetchone()
                if row:
                    version = row[1] + 1
                    old_value = row[0]
                    self.conn.execute(
                        "UPDATE config SET value=?, version=? WHERE key=?",
                        (value, version, key),
                    )
                else:
                    version = 1
                    old_value = ""
                    self.conn.execute(
                        "INSERT INTO config(key, value, version) VALUES (?, ?, ?)",
                        (key, value, version),
                    )
                ts = datetime.utcnow().isoformat()
                self.conn.execute(
                    "INSERT INTO audit(key, old_value, new_value, version, user, reason, ts) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (key, old_value, value, version, user, reason, ts),
                )
                self.conn.commit()
                return {"key": key, "value": value, "version": version}

            entry = await asyncio.to_thread(_set)

        # Broadcast to subscribers
        for q in list(self.subscribers):
            await q.put(entry)
        return entry

    def register(self) -> asyncio.Queue[Dict[str, Any]]:
        q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.subscribers.append(q)
        return q


STORE = SQLiteConfigStore(os.getenv("CONFIG_DB", "config.db"))

# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------

AUDIT_LOG = Path(os.getenv("CONFIG_AUDIT_LOG", "config_audit.log"))
AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)

audit_logger = logging.getLogger("config_audit")
if not audit_logger.handlers:
    audit_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(AUDIT_LOG)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    handler.setFormatter(formatter)
    audit_logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------

API_KEYS: Dict[str, Dict[str, str]] = {
    # api_key: {"user": str, "role": str}
    os.getenv("CONFIG_ADMIN_KEY", "admin-key"): {"user": "admin", "role": "admin"},
    os.getenv("CONFIG_READER_KEY", "reader-key"): {"user": "reader", "role": "reader"},
}

ACL = {
    "admin": {"read": True, "write": True},
    "reader": {"read": True, "write": False},
}

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def require_permission(write: bool):
    async def _deps(key: str = Security(api_key_header)) -> str:
        info = API_KEYS.get(key)
        if not info:
            raise HTTPException(status_code=401, detail="Unauthorized")
        perms = ACL.get(info["role"], {})
        allowed = perms.get("write" if write else "read", False)
        if not allowed:
            raise HTTPException(status_code=403, detail="Forbidden")
        return info["user"]

    return _deps


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------


@dataclass
class UpdateModel:
    key: str
    value: Any
    reason: str


app = FastAPI(title="Config Service")


@app.get("/config/{key}")
async def get_config(key: str, user: str = Depends(require_permission(False))):
    entry = await STORE.get(key)
    if not entry:
        raise HTTPException(status_code=404, detail="Not found")
    return entry


@app.post("/config")
async def update_config(model: UpdateModel, user: str = Depends(require_permission(True))):
    entry = await STORE.set(model.key, json.dumps(model.value), user, model.reason)
    audit_logger.info(
        "user=%s key=%s value=%s version=%s reason=%s",
        user,
        model.key,
        model.value,
        entry["version"],
        model.reason,
    )
    return {"success": True, **entry}


# ---------------------------------------------------------------------------
# gRPC service
# ---------------------------------------------------------------------------


async def _authorize_rpc(context: grpc.aio.ServicerContext, write: bool) -> str:
    api_key = None
    for key, value in context.invocation_metadata():
        if key == "x-api-key":
            api_key = value
            break
    info = API_KEYS.get(api_key or "")
    if not info:
        await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Unauthorized")
    perms = ACL.get(info["role"], {})
    allowed = perms.get("write" if write else "read", False)
    if not allowed:
        await context.abort(grpc.StatusCode.PERMISSION_DENIED, "Forbidden")
    return info["user"]


class ConfigServicer(config_pb2_grpc.ConfigServiceServicer):
    async def GetConfig(self, request: config_pb2.KeyRequest, context):
        await _authorize_rpc(context, write=False)
        entry = await STORE.get(request.key)
        if not entry:
            await context.abort(grpc.StatusCode.NOT_FOUND, "Key not found")
        return config_pb2.ConfigEntry(
            key=entry["key"], value=entry["value"], version=entry["version"]
        )

    async def UpdateConfig(self, request: config_pb2.UpdateRequest, context):
        user = await _authorize_rpc(context, write=True)
        entry = await STORE.set(request.key, request.value, user, request.reason)
        audit_logger.info(
            "user=%s key=%s value=%s version=%s reason=%s",
            user,
            request.key,
            request.value,
            entry["version"],
            request.reason,
        )
        return config_pb2.UpdateResponse(success=True, version=entry["version"])

    async def WatchConfig(self, request: empty_pb2.Empty, context):
        await _authorize_rpc(context, write=False)
        queue = STORE.register()
        while True:
            entry = await queue.get()
            yield config_pb2.ConfigEntry(
                key=entry["key"], value=entry["value"], version=entry["version"]
            )


async def serve_grpc(address: str = "[::]:50052") -> None:
    """Start the gRPC server."""
    server = grpc.aio.server()
    config_pb2_grpc.add_ConfigServiceServicer_to_server(ConfigServicer(), server)
    server.add_insecure_port(address)
    await server.start()
    await server.wait_for_termination()


# ---------------------------------------------------------------------------
# Helpers for local subscribers
# ---------------------------------------------------------------------------


async def subscribe_to_changes() -> AsyncIterator[Dict[str, Any]]:
    """Yield configuration changes as they happen."""
    q = STORE.register()
    while True:
        yield await q.get()

