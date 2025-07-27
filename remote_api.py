from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Security,
    WebSocket,
    WebSocketDisconnect,
    Query,
)
from fastapi.security.api_key import APIKeyHeader
from subprocess import Popen
from typing import Dict, Any, Set
import asyncio
from pydantic import BaseModel
import os
from utils import update_config
from log_utils import LOG_FILE


class ConfigUpdate(BaseModel):
    key: str
    value: Any
    reason: str


app = FastAPI(title="MT5 Bot Controller")

API_KEY = os.getenv("API_KEY", "")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def authorize(key: str = Security(api_key_header)) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

bots: Dict[str, Popen] = {}
metrics_clients: Set[WebSocket] = set()

async def broadcast_update(data: Dict[str, Any]) -> None:
    """Send ``data`` to all connected WebSocket clients."""
    dead: Set[WebSocket] = set()
    for ws in metrics_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    for ws in dead:
        metrics_clients.discard(ws)


def _log_tail(lines: int) -> str:
    """Return the last `lines` from the shared log file."""
    if not LOG_FILE.exists():
        return ""
    with open(LOG_FILE, "r") as f:
        return "".join(f.readlines()[-lines:])

@app.get("/bots")
async def list_bots(_: None = Depends(authorize)):
    """Return running bot IDs and their status."""
    return {bid: proc.poll() is None for bid, proc in bots.items()}

@app.post("/bots/{bot_id}/start")
async def start_bot(bot_id: str, _: None = Depends(authorize)):
    """Launch a realtime training instance for the given bot."""
    if bot_id in bots and bots[bot_id].poll() is None:
        raise HTTPException(status_code=400, detail="Bot already running")
    proc = Popen(["python", "realtime_train.py"])
    bots[bot_id] = proc
    return {"bot": bot_id, "status": "started"}

@app.post("/bots/{bot_id}/stop")
async def stop_bot(bot_id: str, _: None = Depends(authorize)):
    """Terminate a running bot."""
    proc = bots.get(bot_id)
    if not proc:
        raise HTTPException(status_code=404, detail="Bot not found")
    proc.terminate()
    bots.pop(bot_id)
    return {"bot": bot_id, "status": "stopped"}


@app.post("/metrics")
async def push_metrics(data: Dict[str, Any], _: None = Depends(authorize)):
    """Receive metrics data and broadcast to clients."""
    await broadcast_update(data)
    return {"status": "ok"}


@app.websocket("/ws/metrics")
async def metrics_ws(websocket: WebSocket, api_key: str = Query("")):
    """WebSocket endpoint for streaming metrics."""
    if API_KEY and api_key != API_KEY:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    metrics_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        metrics_clients.discard(websocket)


@app.get("/logs")
async def get_logs(lines: int = 50, _: None = Depends(authorize)):
    """Return the last N lines from the bot log file."""
    if not LOG_FILE.exists():
        raise HTTPException(status_code=404, detail="Log file not found")
    with open(LOG_FILE, "r") as f:
        data = f.readlines()[-lines:]
    return {"logs": "".join(data)}


@app.post("/config")
async def update_configuration(change: ConfigUpdate, _: None = Depends(authorize)):
    """Update a configuration variable."""
    update_config(change.key, change.value, change.reason)
    return {"status": "updated", change.key: change.value}


@app.get("/health")
async def health(lines: int = 20, _: None = Depends(authorize)):
    """Return service status and recent log lines."""
    return {
        "running": True,
        "bots": {
            bid: {
                "running": proc.poll() is None,
                "pid": proc.pid,
                "returncode": proc.returncode,
            }
            for bid, proc in bots.items()
        },
        "logs": _log_tail(lines),
    }


@app.get("/bots/{bot_id}/status")
async def bot_status(bot_id: str, lines: int = 20, _: None = Depends(authorize)):
    """Return bot state and recent log lines."""
    proc = bots.get(bot_id)
    if not proc:
        raise HTTPException(status_code=404, detail="Bot not found")
    return {
        "bot": bot_id,
        "running": proc.poll() is None,
        "pid": proc.pid,
        "returncode": proc.returncode,
        "logs": _log_tail(lines),
    }
