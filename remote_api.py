from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Security,
    WebSocket,
    WebSocketDisconnect,
    Query,
    Request,
)
from fastapi.security.api_key import APIKeyHeader
from subprocess import Popen
from typing import Dict, Any, Set, Optional
import asyncio
from pydantic import BaseModel
import os
from utils.secret_manager import SecretManager
from pathlib import Path
import uvicorn
from utils import update_config
from utils.graceful_exit import graceful_exit
import socket
import time
import datetime
import csv

try:
    from utils.alerting import send_alert
except Exception:  # pragma: no cover - utils may be stubbed in tests

    def send_alert(msg: str) -> None:  # type: ignore
        return


from log_utils import LOG_FILE, setup_logging
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from dataclasses import dataclass
from risk_manager import risk_manager
from scheduler import start_scheduler

try:
    from utils.resource_monitor import ResourceMonitor
except Exception:  # pragma: no cover - utils may be stubbed in tests

    class ResourceMonitor:  # type: ignore
        def __init__(self, *a, **k):
            self.max_rss_mb = None
            self.max_cpu_pct = None

        def start(self) -> None:
            return


from metrics import RESOURCE_RESTARTS


class ConfigUpdate(BaseModel):
    key: str
    value: Any
    reason: str


app = FastAPI(title="MT5 Bot Controller")
logger = setup_logging()

MAX_RSS_MB = float(os.getenv("MAX_RSS_MB", "0") or 0)
MAX_CPU_PCT = float(os.getenv("MAX_CPU_PCT", "0") or 0)
resource_watchdog = ResourceMonitor(
    max_rss_mb=MAX_RSS_MB or None, max_cpu_pct=MAX_CPU_PCT or None
)

WATCHDOG_USEC = int(os.getenv("WATCHDOG_USEC", "0") or 0)


def _sd_notify(msg: str) -> None:
    """Send a notification to systemd if the notify socket is available."""
    sock_path = os.getenv("NOTIFY_SOCKET")
    if not sock_path:
        return
    if sock_path.startswith("@"):
        sock_path = "\0" + sock_path[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(sock_path)
            s.send(msg.encode())
    except Exception:  # pragma: no cover - best effort only
        pass


async def _systemd_watchdog(interval: float) -> None:
    """Periodically notify systemd that the service is healthy."""
    while True:
        await asyncio.sleep(interval)
        _sd_notify("WATCHDOG=1")


if os.name != "nt" and getattr(resource_watchdog, "capabilities", None):
    if resource_watchdog.capabilities.cpus > 1:  # pragma: no cover - host dependent
        try:
            import uvloop  # type: ignore

            uvloop.install()
            logger.info("uvloop installed")
        except Exception:  # pragma: no cover - uvloop optional
            logger.info("uvloop not available; using default event loop")


async def _handle_resource_breach(reason: str) -> None:
    logger.error("Resource watchdog triggered: %s", reason)
    RESOURCE_RESTARTS.inc()
    send_alert(f"Resource watchdog triggered: {reason}")
    await graceful_exit()


API_KEY = SecretManager().get_secret("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY secret is required")

CERT_FILE = Path("certs/api.crt")
KEY_FILE = Path("certs/api.key")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


async def authorize(key: str = Security(api_key_header)) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


bots_lock = asyncio.Lock()

RATE_LIMIT = int(os.getenv("RATE_LIMIT", "5"))
BUCKET_TTL = int(os.getenv("BUCKET_TTL", str(15 * 60)))
AUDIT_LOG = Path("logs/api_audit.csv")
AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class TokenBucket:
    tokens: float
    last: float
    last_seen: float


_buckets: Dict[str, TokenBucket] = {}


def _allow_request(key: str) -> bool:
    now = time.time()
    for k, b in list(_buckets.items()):
        if now - b.last_seen > BUCKET_TTL:
            _buckets.pop(k, None)
    bucket = _buckets.get(key)
    if not bucket:
        bucket = TokenBucket(tokens=RATE_LIMIT, last=now, last_seen=now)
        _buckets[key] = bucket
    else:
        bucket.tokens = min(
            RATE_LIMIT, bucket.tokens + (now - bucket.last) * RATE_LIMIT
        )
        bucket.last = now
        bucket.last_seen = now
    if bucket.tokens < 1:
        return False
    bucket.tokens -= 1
    return True


def _audit_log(key: str, action: str, status: int) -> None:
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG, "a", newline="") as f:
        csv.writer(f).writerow(
            [datetime.datetime.utcnow().isoformat(), key, action, status]
        )


@app.middleware("http")
async def _rate_limiter(request: Request, call_next):
    client = request.client.host if request.client else "unknown"
    key = request.headers.get("x-api-key") or client
    action = request.url.path
    if not _allow_request(key):
        _audit_log(key, action, 429)
        return Response("Too Many Requests", status_code=429)
    try:
        response = await call_next(request)
    except HTTPException as exc:
        _audit_log(key, action, exc.status_code)
        raise
    except Exception:
        _audit_log(key, action, 500)
        raise
    _audit_log(key, action, response.status_code)
    return response


@dataclass
class BotInfo:
    proc: Popen
    restart_count: int = 0
    exit_code: Optional[int] = None


bots: Dict[str, BotInfo] = {}
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


async def _check_bots_once() -> None:
    """Inspect running bots and restart or remove if exited."""
    async with bots_lock:
        for bid, info in list(bots.items()):
            rc = info.proc.poll()
            if rc is not None:
                info.exit_code = rc
                logger.warning("Bot %s exited with code %s", bid, rc)
                try:
                    info.proc = Popen(["python", "realtime_train.py"])
                    info.restart_count += 1
                except Exception:
                    bots.pop(bid, None)


async def _bot_watcher() -> None:
    """Background task to monitor bot processes."""
    while True:
        await asyncio.sleep(1)
        await _check_bots_once()


@app.on_event("startup")
async def _start_watcher() -> None:
    start_scheduler()
    asyncio.create_task(_bot_watcher())
    if resource_watchdog.max_rss_mb or resource_watchdog.max_cpu_pct:
        resource_watchdog.alert_callback = _handle_resource_breach
        resource_watchdog.start()
    _sd_notify("READY=1")
    if WATCHDOG_USEC:
        interval = WATCHDOG_USEC / 2 / 1_000_000
        asyncio.create_task(_systemd_watchdog(interval))


@app.get("/bots")
async def list_bots(_: None = Depends(authorize)):
    """Return running bot IDs and their status."""
    async with bots_lock:
        return {
            bid: {
                "running": info.proc.poll() is None,
                "exit_code": info.exit_code,
                "restart_count": info.restart_count,
            }
            for bid, info in bots.items()
        }


@app.post("/bots/{bot_id}/start")
async def start_bot(bot_id: str, _: None = Depends(authorize)):
    """Launch a realtime training instance for the given bot."""
    async with bots_lock:
        if bot_id in bots and bots[bot_id].proc.poll() is None:
            raise HTTPException(status_code=400, detail="Bot already running")
        proc = Popen(["python", "realtime_train.py"])
        bots[bot_id] = BotInfo(proc=proc)
    return {"bot": bot_id, "status": "started"}


@app.post("/bots/{bot_id}/stop")
async def stop_bot(bot_id: str, _: None = Depends(authorize)):
    """Terminate a running bot."""
    async with bots_lock:
        info = bots.get(bot_id)
        if not info:
            raise HTTPException(status_code=404, detail="Bot not found")
        info.proc.terminate()
        bots.pop(bot_id)
    return {"bot": bot_id, "status": "stopped"}


@app.post("/metrics")
async def push_metrics(data: Dict[str, Any], _: None = Depends(authorize)):
    """Receive metrics data and broadcast to clients."""
    await broadcast_update(data)
    return {"status": "ok"}


@app.get("/metrics")
async def get_metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/risk/status")
async def risk_status(_: None = Depends(authorize)):
    """Return aggregated risk metrics."""
    return risk_manager.status()


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


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    """Liveness probe for systemd or external monitors."""
    return {"status": "ok"}


@app.get("/health")
async def health(lines: int = 20, _: None = Depends(authorize)):
    """Return service status and recent log lines."""
    async with bots_lock:
        bot_data = {
            bid: {
                "running": info.proc.poll() is None,
                "pid": info.proc.pid,
                "returncode": info.proc.returncode,
                "exit_code": info.exit_code,
                "restart_count": info.restart_count,
            }
            for bid, info in bots.items()
        }
    return {
        "running": True,
        "bots": bot_data,
        "logs": _log_tail(lines),
    }


@app.get("/bots/{bot_id}/status")
async def bot_status(bot_id: str, lines: int = 20, _: None = Depends(authorize)):
    """Return bot state and recent log lines."""
    async with bots_lock:
        info = bots.get(bot_id)
        if not info:
            raise HTTPException(status_code=404, detail="Bot not found")
        data = {
            "bot": bot_id,
            "running": info.proc.poll() is None,
            "pid": info.proc.pid,
            "returncode": info.proc.returncode,
            "exit_code": info.exit_code,
            "restart_count": info.restart_count,
            "logs": _log_tail(lines),
        }
    return data


@app.get("/bots/{bot_id}/logs")
async def bot_logs(bot_id: str, lines: int = 50, _: None = Depends(authorize)):
    """Return the last N log lines for a bot."""
    async with bots_lock:
        if bot_id not in bots:
            raise HTTPException(status_code=404, detail="Bot not found")
    return {"bot": bot_id, "logs": _log_tail(lines)}


if __name__ == "__main__":
    if not CERT_FILE.exists() or not KEY_FILE.exists():
        raise RuntimeError("SSL certificate or key file missing")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_certfile=str(CERT_FILE),
        ssl_keyfile=str(KEY_FILE),
    )
