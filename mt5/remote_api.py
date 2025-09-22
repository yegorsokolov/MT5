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
from typing import Dict, Any, Set, Optional, Deque, Awaitable, Callable
import asyncio
from pydantic import BaseModel
import os
import logging
from logging.handlers import RotatingFileHandler
from utils.secret_manager import SecretManager
from pathlib import Path
import uvicorn
from utils import update_config
from utils.graceful_exit import graceful_exit
import socket
import time
import datetime
import hmac
import hashlib
from collections import deque
import contextlib

try:
    from utils.alerting import send_alert
except Exception:  # pragma: no cover - utils may be stubbed in tests

    def send_alert(msg: str) -> None:  # type: ignore
        return
from mt5.log_utils import LOG_FILE, setup_logging
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Gauge, REGISTRY
from fastapi.responses import Response
from dataclasses import dataclass, field
from mt5 import risk_manager as risk_manager_module
from mt5.risk_manager import risk_manager
from mt5.scheduler import (
    start_scheduler,
    stop_scheduler,
    schedule_retrain,
    resource_reprobe,
    run_drift_detection,
    run_feature_importance_drift,
    run_change_point_detection,
    cleanup_checkpoints,
    rebuild_news_vectors,
    run_decision_review,
    run_trade_analysis,
    run_diagnostics,
    update_regime_performance,
    run_backups,
)

try:
    from utils.resource_monitor import ResourceMonitor
except Exception:  # pragma: no cover - utils may be stubbed in tests

    class ResourceMonitor:  # type: ignore
        def __init__(self, *a, **k):
            self.max_rss_mb = None
            self.max_cpu_pct = None
            self.started = False
            self.capabilities = None

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.started = False
from mt5.metrics import RESOURCE_RESTARTS

try:
    from mt5.metrics import BOT_BACKOFFS, BOT_RESTARTS, BOT_RESTART_FAILURES
except Exception:  # pragma: no cover - optional metrics
    BOT_BACKOFFS = None  # type: ignore
    BOT_RESTARTS = None  # type: ignore
    BOT_RESTART_FAILURES = None  # type: ignore


class ConfigUpdate(BaseModel):
    key: str
    value: Any
    reason: str


class ControlRequest(BaseModel):
    task: str


class RetrainRequest(BaseModel):
    model: str = "classic"
    update_hyperparams: bool = False


logger: logging.Logger = logging.getLogger(__name__)
_logging_initialized: bool = False


def init_logging() -> logging.Logger:
    """Initialise application logging once and return the configured logger."""

    global logger, _logging_initialized
    if not _logging_initialized:
        logger = setup_logging()
        _logging_initialized = True
    return logger


app = FastAPI(title="MT5 Bot Controller")

MAX_RSS_MB = float(os.getenv("MAX_RSS_MB", "0") or 0)
MAX_CPU_PCT = float(os.getenv("MAX_CPU_PCT", "0") or 0)
resource_watchdog = ResourceMonitor(
    max_rss_mb=MAX_RSS_MB or None, max_cpu_pct=MAX_CPU_PCT or None
)

WATCHDOG_USEC = int(os.getenv("WATCHDOG_USEC", "0") or 0)


BOT_BACKOFF_BASE_SECONDS = max(float(os.getenv("BOT_BACKOFF_BASE", "1") or 1), 0.0)
BOT_BACKOFF_MAX_SECONDS = max(
    float(os.getenv("BOT_BACKOFF_MAX", "60") or 60), BOT_BACKOFF_BASE_SECONDS
)
BOT_BACKOFF_RESET_SECONDS = max(
    float(os.getenv("BOT_BACKOFF_RESET", "300") or 300), 0.0
)
BOT_MAX_CRASHES = max(int(os.getenv("BOT_MAX_CRASHES", "5") or 5), 1)
BOT_CRASH_WINDOW = max(
    float(os.getenv("BOT_CRASH_WINDOW", "600") or 600), BOT_BACKOFF_BASE_SECONDS
)
RESTART_HISTORY_LIMIT = max(BOT_MAX_CRASHES * 2, 10)


async def _run_sync(func: Callable[[], None]) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, func)


CONTROL_TASKS: Dict[str, Callable[[], Awaitable[Any]]] = {
    "cleanup_checkpoints": lambda: _run_sync(cleanup_checkpoints),
    "run_drift_detection": lambda: _run_sync(run_drift_detection),
    "run_feature_importance_drift": lambda: _run_sync(run_feature_importance_drift),
    "run_change_point_detection": lambda: _run_sync(run_change_point_detection),
    "run_trade_analysis": lambda: _run_sync(run_trade_analysis),
    "run_decision_review": lambda: _run_sync(run_decision_review),
    "run_diagnostics": lambda: _run_sync(run_diagnostics),
    "rebuild_news_vectors": lambda: _run_sync(rebuild_news_vectors),
    "update_regime_performance": lambda: _run_sync(update_regime_performance),
    "run_backups": lambda: _run_sync(run_backups),
}
CONTROL_TASKS["resource_reprobe"] = resource_reprobe

RETRAIN_MODELS = ("classic", "nn", "rl")


def _start_risk_background_services() -> None:
    starter = getattr(risk_manager_module, "ensure_scheduler_started", None)
    if callable(starter):
        try:
            starter()
            return
        except Exception:
            logger.exception(
                "Risk manager initialization helper failed; falling back to direct start"
            )
    start_scheduler()


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
    try:
        while True:
            await asyncio.sleep(interval)
            _sd_notify("WATCHDOG=1")
    except asyncio.CancelledError:
        return


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


CERT_FILE = Path("certs/api.crt")
KEY_FILE = Path("certs/api.key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

API_KEY: Optional[str] = None
AUDIT_SECRET: Optional[str] = None


def _resolve_secret(
    name: str,
    *,
    override: Optional[str] = None,
    secret_manager: Optional[SecretManager] = None,
) -> str:
    if override:
        return override
    env_value = os.getenv(name)
    if env_value:
        return env_value
    manager = secret_manager or SecretManager()
    value = manager.get_secret(name)
    if not value:
        raise RuntimeError(f"{name} secret is required")
    return value


def init_remote_api(
    *,
    secret_manager: Optional[SecretManager] = None,
    api_key: Optional[str] = None,
    audit_secret: Optional[str] = None,
) -> None:
    """Resolve and cache secrets required by the API."""

    init_logging()

    global API_KEY, AUDIT_SECRET

    API_KEY = _resolve_secret(
        "API_KEY", override=api_key, secret_manager=secret_manager
    )
    AUDIT_SECRET = _resolve_secret(
        "AUDIT_LOG_SECRET", override=audit_secret, secret_manager=secret_manager
    )


async def authorize(key: str = Security(api_key_header)) -> None:
    if API_KEY is None:
        raise HTTPException(status_code=503, detail="API not initialized")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


bots_lock = asyncio.Lock()

RATE_LIMIT = int(os.getenv("RATE_LIMIT", "5"))
BUCKET_TTL = int(os.getenv("BUCKET_TTL", str(15 * 60)))
AUDIT_LOG = Path("logs/api_audit.csv")
AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)

_audit_handler = RotatingFileHandler(AUDIT_LOG, maxBytes=1_000_000, backupCount=5)
_audit_handler.setFormatter(logging.Formatter("%(message)s"))
audit_logger = logging.getLogger("api_audit")
audit_logger.setLevel(logging.INFO)
audit_logger.addHandler(_audit_handler)


@dataclass
class TokenBucket:
    tokens: float
    last: float
    last_seen: float


_buckets: Dict[str, TokenBucket] = {}

try:
    API_RATE_LIMIT_REMAINING = Gauge(
        "api_rate_limit_remaining", "Remaining API rate limit tokens", ["key"]
    )
except ValueError:
    API_RATE_LIMIT_REMAINING = REGISTRY._names_to_collectors.get(
        "api_rate_limit_remaining"
    )
except Exception:
    API_RATE_LIMIT_REMAINING = None


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
        if API_RATE_LIMIT_REMAINING:
            API_RATE_LIMIT_REMAINING.labels(key=key).set(bucket.tokens)
        return False
    bucket.tokens -= 1
    if API_RATE_LIMIT_REMAINING:
        API_RATE_LIMIT_REMAINING.labels(key=key).set(bucket.tokens)
    return True


def _audit_log(key: str, action: str, status: int) -> None:
    if AUDIT_SECRET is None:
        raise RuntimeError("AUDIT_LOG_SECRET is not initialized")
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().isoformat()
    line = f"{ts},{key},{action},{status}"
    sig = hmac.new(AUDIT_SECRET.encode(), line.encode(), hashlib.sha256).hexdigest()
    audit_logger.info("%s,%s", line, sig)


@app.middleware("http")
async def _rate_limiter(request: Request, call_next):
    client = request.client.host if request.client else "unknown"
    key = (
        request.headers.get("X-API-Key")
        or request.headers.get("x-api-key")
        or client
    )
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
    last_start: float = field(default_factory=time.monotonic)
    last_exit: Optional[float] = None
    cooldown_until: float = 0.0
    pending_restart: bool = False
    failure_streak: int = 0
    backoff_logged: bool = False
    restart_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=RESTART_HISTORY_LIMIT)
    )


bots: Dict[str, BotInfo] = {}
metrics_clients: Set[WebSocket] = set()
_background_tasks: Set[asyncio.Task[Any]] = set()
_resource_watchdog_running = False


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


def _register_background_task(coro: Awaitable[Any]) -> asyncio.Task[Any]:
    """Track background tasks so they can be cancelled during shutdown."""

    task = asyncio.create_task(coro)

    def _cleanup(completed: asyncio.Task[Any]) -> None:
        _background_tasks.discard(completed)

    _background_tasks.add(task)
    task.add_done_callback(_cleanup)
    return task


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
            if rc is None:
                if (
                    info.failure_streak
                    and info.last_start
                    and (time.monotonic() - info.last_start) >= BOT_BACKOFF_RESET_SECONDS
                ):
                    info.failure_streak = 0
                continue

            now = time.monotonic()
            info.exit_code = rc
            info.last_exit = now

            if not info.pending_restart:
                logger.warning("Bot %s exited with code %s", bid, rc)
                while info.restart_history and now - info.restart_history[0] > BOT_CRASH_WINDOW:
                    info.restart_history.popleft()
                info.restart_history.append(now)
                recent_attempts = len(info.restart_history)
                if recent_attempts > BOT_MAX_CRASHES:
                    bots.pop(bid, None)
                    message = (
                        f"Bot {bid} exceeded restart limit after "
                        f"{recent_attempts - 1} attempts; last exit code {rc}"
                    )
                    logger.error(message)
                    try:
                        send_alert(message)
                    except Exception:
                        logger.exception("send_alert failed for %s", bid)
                    if BOT_RESTART_FAILURES:
                        try:
                            BOT_RESTART_FAILURES.labels(bot=bid).inc()
                        except Exception:
                            pass
                    continue
                if (
                    info.last_start
                    and (now - info.last_start) >= BOT_BACKOFF_RESET_SECONDS
                ):
                    info.failure_streak = 0
                info.failure_streak += 1
                delay = min(
                    BOT_BACKOFF_MAX_SECONDS,
                    BOT_BACKOFF_BASE_SECONDS * (2 ** (info.failure_streak - 1)),
                )
                info.cooldown_until = now + delay
                info.pending_restart = True
                info.backoff_logged = False
                if BOT_BACKOFFS:
                    try:
                        BOT_BACKOFFS.labels(bot=bid).inc()
                    except Exception:
                        pass

            if not info.pending_restart:
                continue

            remaining = info.cooldown_until - time.monotonic()
            if remaining > 0:
                if not info.backoff_logged and info.cooldown_until > 0:
                    logger.info(
                        "Restarting bot %s in %.1fs (attempt %s)",
                        bid,
                        max(remaining, 0.0),
                        info.failure_streak,
                    )
                    info.backoff_logged = True
                continue

            try:
                new_proc = Popen(["python", "-m", "mt5.realtime_train"])
            except Exception:
                bots.pop(bid, None)
                logger.exception("Failed to restart bot %s", bid)
                message = f"Failed to restart bot {bid}; removing from registry"
                try:
                    send_alert(message)
                except Exception:
                    logger.exception("send_alert failed for %s", bid)
                if BOT_RESTART_FAILURES:
                    try:
                        BOT_RESTART_FAILURES.labels(bot=bid).inc()
                    except Exception:
                        pass
                continue

            info.proc = new_proc
            info.restart_count += 1
            info.last_start = time.monotonic()
            info.pending_restart = False
            info.cooldown_until = 0.0
            info.backoff_logged = False
            if BOT_RESTARTS:
                try:
                    BOT_RESTARTS.labels(bot=bid).inc()
                except Exception:
                    pass


async def _bot_watcher() -> None:
    """Background task to monitor bot processes."""
    try:
        while True:
            await asyncio.sleep(1)
            await _check_bots_once()
    except asyncio.CancelledError:
        return


@app.on_event("startup")
async def _start_watcher() -> None:
    init_remote_api()
    _start_risk_background_services()
    _register_background_task(_bot_watcher())
    global _resource_watchdog_running
    if (
        (resource_watchdog.max_rss_mb or resource_watchdog.max_cpu_pct)
        and not _resource_watchdog_running
    ):
        resource_watchdog.alert_callback = _handle_resource_breach
        try:
            resource_watchdog.start()
            _resource_watchdog_running = True
        except Exception:
            logger.exception("Failed to start resource watchdog")
    _sd_notify("READY=1")
    if WATCHDOG_USEC:
        interval = WATCHDOG_USEC / 2 / 1_000_000
        _register_background_task(_systemd_watchdog(interval))


@app.on_event("shutdown")
async def _stop_scheduler_event() -> None:
    stop_scheduler()
    tasks = list(_background_tasks)
    for task in tasks:
        task.cancel()
    for task in tasks:
        with contextlib.suppress(asyncio.CancelledError):
            await task
    _background_tasks.difference_update(tasks)
    global _resource_watchdog_running
    if _resource_watchdog_running:
        try:
            resource_watchdog.stop()
        except Exception:
            logger.exception("Failed to stop resource watchdog")
        _resource_watchdog_running = False


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
        proc = Popen(["python", "-m", "mt5.realtime_train"])
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


@app.get("/controls")
async def list_controls(_: None = Depends(authorize)):
    """Return available manual control tasks."""

    return {
        "tasks": sorted(CONTROL_TASKS.keys()),
        "retrain_models": sorted(RETRAIN_MODELS),
    }


@app.post("/controls/run")
async def run_control(req: ControlRequest, _: None = Depends(authorize)):
    """Execute a manual control task."""

    func = CONTROL_TASKS.get(req.task)
    if func is None:
        raise HTTPException(status_code=404, detail="Unknown control task")
    try:
        await func()
    except Exception as exc:
        logger.exception("Manual control %s failed", req.task)
        raise HTTPException(status_code=500, detail=f"Task {req.task} failed: {exc}") from exc
    return {"status": "ok", "task": req.task}


@app.post("/controls/retrain")
async def schedule_manual_retrain(req: RetrainRequest, _: None = Depends(authorize)):
    """Schedule a manual model retraining."""

    model = (req.model or "classic").strip()
    if not model:
        model = "classic"
    normalised = model.lower()
    if normalised not in {m.lower() for m in RETRAIN_MODELS}:
        raise HTTPException(status_code=400, detail="Unsupported model")
    try:
        schedule_retrain(model=normalised, update_hyperparams=req.update_hyperparams)
    except Exception as exc:
        logger.exception("Manual retrain scheduling failed for %s", normalised)
        raise HTTPException(status_code=500, detail=f"Retrain scheduling failed: {exc}") from exc
    return {"status": "scheduled", "model": normalised}


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
    init_remote_api()
    if not CERT_FILE.exists() or not KEY_FILE.exists():
        raise RuntimeError("SSL certificate or key file missing")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_certfile=str(CERT_FILE),
        ssl_keyfile=str(KEY_FILE),
    )
