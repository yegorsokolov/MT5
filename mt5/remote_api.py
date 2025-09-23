"""Local management helpers that replace the archived FastAPI service.

The original project exposed process control, configuration updates and
observability hooks through ``archive/bot_apis/remote_api.py``.  This module
re-creates those capabilities as in-process async helpers so the bot can be
controlled without hosting an HTTP service.  The functions intentionally keep
names and semantics compatible with the historical API so existing tooling (for
example the gRPC bridge) can continue to call them.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as _dt
import hmac
import hashlib
import importlib
import importlib.util
import inspect
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import socket
from subprocess import Popen, TimeoutExpired
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Deque, Dict, Optional, Set

from fastapi import HTTPException
from mt5.log_utils import LOG_FILE, setup_logging
from mt5 import risk_manager as risk_manager_module
from mt5.risk_manager import risk_manager
from mt5.scheduler import (
    cleanup_checkpoints,
    rebuild_news_vectors,
    resource_reprobe,
    run_backups,
    run_change_point_detection,
    run_decision_review,
    run_diagnostics,
    run_drift_detection,
    run_feature_importance_drift,
    run_trade_analysis,
    schedule_retrain,
    start_scheduler,
    stop_scheduler,
    update_regime_performance,
)
from mt5.metrics import BOT_BACKOFFS, BOT_RESTARTS, BOT_RESTART_FAILURES, RESOURCE_RESTARTS
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from utils import update_config
from utils.graceful_exit import graceful_exit
from utils.secret_manager import SecretManager

_alerting_spec = importlib.util.find_spec("utils.alerting")
if _alerting_spec is not None:
    _alerting_module = importlib.import_module("utils.alerting")
    send_alert: Callable[[str], None] = getattr(_alerting_module, "send_alert", lambda msg: None)
else:  # pragma: no cover - fallback for stripped deployments
    def send_alert(msg: str) -> None:
        return None

_resource_monitor_spec = importlib.util.find_spec("utils.resource_monitor")
if _resource_monitor_spec is not None:
    _resource_module = importlib.import_module("utils.resource_monitor")
    ResourceMonitor = getattr(_resource_module, "ResourceMonitor")
else:  # pragma: no cover - fallback for stripped deployments
    class ResourceMonitor:  # type: ignore[empty-body]
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.max_rss_mb = kwargs.get("max_rss_mb")
            self.max_cpu_pct = kwargs.get("max_cpu_pct")
            self.alert_callback: Optional[Callable[[str], Optional[Awaitable[None]]]] = None
            self.capabilities = None

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None


logger: logging.Logger = logging.getLogger(__name__)
_logging_initialized = False


def init_logging() -> logging.Logger:
    """Initialise application logging once and return the configured logger."""

    global logger, _logging_initialized
    if not _logging_initialized:
        logger = setup_logging()
        _logging_initialized = True
    return logger


MAX_RSS_MB = float(os.getenv("MAX_RSS_MB", "0") or 0)
MAX_CPU_PCT = float(os.getenv("MAX_CPU_PCT", "0") or 0)
resource_watchdog = ResourceMonitor(max_rss_mb=MAX_RSS_MB or None, max_cpu_pct=MAX_CPU_PCT or None)
WATCHDOG_USEC = int(os.getenv("WATCHDOG_USEC", "0") or 0)

BOT_BACKOFF_BASE_SECONDS = max(float(os.getenv("BOT_BACKOFF_BASE", "1") or 1), 0.0)
BOT_BACKOFF_MAX_SECONDS = max(
    float(os.getenv("BOT_BACKOFF_MAX", "60") or 60), BOT_BACKOFF_BASE_SECONDS
)
BOT_BACKOFF_RESET_SECONDS = max(float(os.getenv("BOT_BACKOFF_RESET", "300") or 300), 0.0)
BOT_MAX_CRASHES = max(int(os.getenv("BOT_MAX_CRASHES", "5") or 5), 1)
BOT_CRASH_WINDOW = max(
    float(os.getenv("BOT_CRASH_WINDOW", "600") or 600), BOT_BACKOFF_BASE_SECONDS
)
RESTART_HISTORY_LIMIT = max(BOT_MAX_CRASHES * 2, 10)

API_KEY: Optional[str] = None
AUDIT_SECRET: Optional[str] = None
_INITIALIZED = False

AUDIT_LOG = Path("logs/api_audit.csv")
AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
_audit_handler = RotatingFileHandler(AUDIT_LOG, maxBytes=1_000_000, backupCount=5)
_audit_handler.setFormatter(logging.Formatter("%(message)s"))
audit_logger = logging.getLogger("api_audit")
audit_logger.setLevel(logging.INFO)
audit_logger.addHandler(_audit_handler)

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


@dataclass
class ConfigUpdate:
    key: str
    value: Any
    reason: str


@dataclass
class ControlRequest:
    task: str


@dataclass
class RetrainRequest:
    model: str = "classic"
    update_hyperparams: bool = False


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
    restart_history: Deque[float] = field(default_factory=lambda: deque(maxlen=RESTART_HISTORY_LIMIT))


bots: Dict[str, BotInfo] = {}
bots_lock = asyncio.Lock()
metrics_clients: Set[Callable[[Dict[str, Any]], Any]] = set()
_background_tasks: Set[asyncio.Task[Any]] = set()
_bot_watcher_task: Optional[asyncio.Task[Any]] = None
_systemd_task: Optional[asyncio.Task[Any]] = None
_task_loop: Optional[asyncio.AbstractEventLoop] = None
_resource_watchdog_running = False


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
        except Exception:  # pragma: no cover - defensive fallback
            logger.exception(
                "Risk manager initialization helper failed; falling back to direct start"
            )
    start_scheduler()


def _sd_notify(msg: str) -> None:
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
    try:
        while True:
            await asyncio.sleep(interval)
            _sd_notify("WATCHDOG=1")
    except asyncio.CancelledError:
        return


async def _handle_resource_breach(reason: str) -> None:
    logger.error("Resource watchdog triggered: %s", reason)
    try:
        RESOURCE_RESTARTS.inc()
    except Exception:  # pragma: no cover - metrics best effort
        pass
    try:
        send_alert(f"Resource watchdog triggered: {reason}")
    except Exception:  # pragma: no cover - alerting best effort
        logger.exception("send_alert failed")
    await graceful_exit()


def _register_background_task(coro: Awaitable[Any]) -> asyncio.Task[Any]:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def _ensure_background_tasks() -> None:
    global _bot_watcher_task, _systemd_task, _task_loop, _resource_watchdog_running
    loop = asyncio.get_running_loop()
    if _task_loop is not None and _task_loop is not loop and _background_tasks:
        await shutdown()
    _task_loop = loop

    if _bot_watcher_task is None or _bot_watcher_task.done():
        _bot_watcher_task = _register_background_task(_bot_watcher())

    if WATCHDOG_USEC and (_systemd_task is None or _systemd_task.done()):
        interval = WATCHDOG_USEC / 2 / 1_000_000
        _systemd_task = _register_background_task(_systemd_watchdog(interval))

    if (resource_watchdog.max_rss_mb or resource_watchdog.max_cpu_pct) and not _resource_watchdog_running:
        resource_watchdog.alert_callback = _handle_resource_breach
        try:
            resource_watchdog.start()
            _resource_watchdog_running = True
        except Exception:  # pragma: no cover - watchdog optional
            logger.exception("Failed to start resource watchdog")


def _log_tail(lines: int) -> str:
    if not LOG_FILE.exists():
        return ""
    with LOG_FILE.open("r", encoding="utf-8", errors="ignore") as fh:
        data = fh.readlines()[-lines:]
    return "".join(data)


async def broadcast_update(data: Dict[str, Any]) -> None:
    dead: Set[Callable[[Dict[str, Any]], Any]] = set()
    pending: list[tuple[Callable[[Dict[str, Any]], Any], asyncio.Task[Any]]] = []
    for cb in list(metrics_clients):
        try:
            result = cb(data)
        except Exception:
            logger.exception("Metrics subscriber failed")
            dead.add(cb)
            continue
        if inspect.isawaitable(result):
            pending.append((cb, asyncio.create_task(result)))
    for cb in dead:
        metrics_clients.discard(cb)
    if not pending:
        return
    results = await asyncio.gather(*(task for _, task in pending), return_exceptions=True)
    for (cb, _), outcome in zip(pending, results, strict=False):
        if isinstance(outcome, Exception):
            logger.exception("Metrics subscriber coroutine failed for %s", getattr(cb, "__name__", repr(cb)))


def register_metrics_consumer(callback: Callable[[Dict[str, Any]], Any]) -> Callable[[], None]:
    metrics_clients.add(callback)

    def _remove() -> None:
        metrics_clients.discard(callback)

    return _remove


async def push_metrics(data: Dict[str, Any]) -> Dict[str, str]:
    await broadcast_update(data)
    return {"status": "ok"}


def collect_metrics() -> bytes:
    return generate_latest()


def metrics_content_type() -> str:
    return CONTENT_TYPE_LATEST


def _audit_log(action: str, status: int, actor: str = "local") -> None:
    if AUDIT_SECRET is None:
        return
    ts = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    line = f"{ts},{actor},{action},{status}"
    sig = hmac.new(AUDIT_SECRET.encode(), line.encode(), hashlib.sha256).hexdigest()
    audit_logger.info("%s,%s", line, sig)


async def _check_bots_once() -> None:
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
                        f"Bot {bid} exceeded restart limit after {recent_attempts - 1} attempts; "
                        f"last exit code {rc}"
                    )
                    logger.error(message)
                    try:
                        send_alert(message)
                    except Exception:  # pragma: no cover
                        logger.exception("send_alert failed for %s", bid)
                    if BOT_RESTART_FAILURES is not None:
                        try:
                            BOT_RESTART_FAILURES.labels(bot=bid).inc()
                        except Exception:
                            pass
                    continue
                if info.last_start and (now - info.last_start) >= BOT_BACKOFF_RESET_SECONDS:
                    info.failure_streak = 0
                info.failure_streak += 1
                delay = min(
                    BOT_BACKOFF_MAX_SECONDS,
                    BOT_BACKOFF_BASE_SECONDS * (2 ** (info.failure_streak - 1)),
                )
                info.cooldown_until = now + delay
                info.pending_restart = True
                info.backoff_logged = False
                if BOT_BACKOFFS is not None:
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
                if BOT_RESTART_FAILURES is not None:
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
            if BOT_RESTARTS is not None:
                try:
                    BOT_RESTARTS.labels(bot=bid).inc()
                except Exception:
                    pass


async def _bot_watcher() -> None:
    try:
        while True:
            await asyncio.sleep(1)
            await _check_bots_once()
    except asyncio.CancelledError:
        return


def init_remote_api(
    *,
    secret_manager: Optional[SecretManager] = None,
    api_key: Optional[str] = None,
    audit_secret: Optional[str] = None,
) -> None:
    """Resolve and cache secrets required by the helpers."""

    init_logging()

    global API_KEY, AUDIT_SECRET, _INITIALIZED
    API_KEY = _resolve_secret("API_KEY", override=api_key, secret_manager=secret_manager)
    AUDIT_SECRET = _resolve_secret(
        "AUDIT_LOG_SECRET", override=audit_secret, secret_manager=secret_manager
    )
    _start_risk_background_services()
    _INITIALIZED = True


def ensure_initialized() -> None:
    if not _INITIALIZED:
        init_remote_api()


def _ensure_ready() -> None:
    if API_KEY is None:
        raise HTTPException(status_code=503, detail="Remote helpers not initialized")


async def list_bots(_: str | None = None) -> Dict[str, Dict[str, Any]]:
    _ensure_ready()
    await _ensure_background_tasks()
    async with bots_lock:
        data = {
            bid: {
                "running": info.proc.poll() is None,
                "exit_code": info.exit_code,
                "restart_count": info.restart_count,
            }
            for bid, info in bots.items()
        }
    _audit_log("list_bots", 200)
    return data


async def start_bot(bot_id: str, actor: str = "system") -> Dict[str, Any]:
    _ensure_ready()
    await _ensure_background_tasks()
    async with bots_lock:
        if bot_id in bots and bots[bot_id].proc.poll() is None:
            raise HTTPException(status_code=400, detail="Bot already running")
        proc = Popen(["python", "-m", "mt5.realtime_train"])
        bots[bot_id] = BotInfo(proc=proc)
    _audit_log(f"start_bot:{bot_id}", 200, actor)
    return {"bot": bot_id, "status": "started"}


async def stop_bot(bot_id: str, actor: str = "system") -> Dict[str, Any]:
    _ensure_ready()
    await _ensure_background_tasks()
    async with bots_lock:
        info = bots.get(bot_id)
        if not info:
            raise HTTPException(status_code=404, detail="Bot not found")
        info.proc.terminate()
        try:
            info.proc.wait(timeout=10)
        except TimeoutExpired:
            info.proc.kill()
        bots.pop(bot_id, None)
    _audit_log(f"stop_bot:{bot_id}", 200, actor)
    return {"bot": bot_id, "status": "stopped"}


async def bot_status(bot_id: str, lines: int = 20) -> Dict[str, Any]:
    _ensure_ready()
    await _ensure_background_tasks()
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
    _audit_log(f"bot_status:{bot_id}", 200)
    return data


async def bot_logs(bot_id: str, lines: int = 50) -> Dict[str, Any]:
    _ensure_ready()
    await _ensure_background_tasks()
    async with bots_lock:
        if bot_id not in bots:
            raise HTTPException(status_code=404, detail="Bot not found")
    _audit_log(f"bot_logs:{bot_id}", 200)
    return {"bot": bot_id, "logs": _log_tail(lines)}


async def get_logs(lines: int = 50) -> Dict[str, str]:
    _ensure_ready()
    await _ensure_background_tasks()
    if not LOG_FILE.exists():
        raise HTTPException(status_code=404, detail="Log file not found")
    _audit_log("logs", 200)
    return {"logs": _log_tail(lines)}


async def update_configuration(change: ConfigUpdate) -> Dict[str, Any]:
    _ensure_ready()
    await _ensure_background_tasks()
    try:
        update_config(change.key, change.value, change.reason)
    except Exception as exc:
        logger.exception("Configuration update failed for %s", change.key)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    _audit_log(f"update_config:{change.key}", 200)
    return {"status": "updated", change.key: change.value}


def list_controls() -> Dict[str, Any]:
    _ensure_ready()
    return {
        "tasks": sorted(CONTROL_TASKS.keys()),
        "retrain_models": sorted(RETRAIN_MODELS),
    }


async def run_control(task: str) -> Dict[str, Any]:
    _ensure_ready()
    await _ensure_background_tasks()
    func = CONTROL_TASKS.get(task)
    if func is None:
        raise HTTPException(status_code=404, detail="Unknown control task")
    try:
        await func()
    except Exception as exc:
        logger.exception("Manual control %s failed", task)
        raise HTTPException(status_code=500, detail=f"Task {task} failed: {exc}") from exc
    _audit_log(f"run_control:{task}", 200)
    return {"status": "ok", "task": task}


async def schedule_manual_retrain(model: str, update_hyperparams: bool = False) -> Dict[str, Any]:
    _ensure_ready()
    await _ensure_background_tasks()
    normalised = (model or "classic").strip().lower() or "classic"
    if normalised not in {m.lower() for m in RETRAIN_MODELS}:
        raise HTTPException(status_code=400, detail="Unsupported model")
    try:
        schedule_retrain(model=normalised, update_hyperparams=update_hyperparams)
    except Exception as exc:
        logger.exception("Manual retrain scheduling failed for %s", normalised)
        raise HTTPException(status_code=500, detail=f"Retrain scheduling failed: {exc}") from exc
    _audit_log(f"schedule_retrain:{normalised}", 200)
    return {"status": "scheduled", "model": normalised}


async def risk_status() -> Dict[str, Any]:
    _ensure_ready()
    await _ensure_background_tasks()
    _audit_log("risk_status", 200)
    return risk_manager.status()


async def health(lines: int = 20) -> Dict[str, Any]:
    _ensure_ready()
    await _ensure_background_tasks()
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
    _audit_log("health", 200)
    return {
        "running": True,
        "bots": bot_data,
        "logs": _log_tail(lines),
    }


async def shutdown() -> None:
    global _resource_watchdog_running, _bot_watcher_task, _systemd_task, _task_loop
    tasks = list(_background_tasks)
    for task in tasks:
        task.cancel()
    for task in tasks:
        with contextlib.suppress(asyncio.CancelledError):
            await task
    _background_tasks.clear()
    _bot_watcher_task = None
    _systemd_task = None
    _task_loop = None
    if _resource_watchdog_running:
        try:
            resource_watchdog.stop()
        except Exception:
            logger.exception("Failed to stop resource watchdog")
        _resource_watchdog_running = False
    try:
        stop_scheduler()
    except Exception:
        logger.exception("Failed to stop scheduler")
    _audit_log("shutdown", 200)


__all__ = [
    name
    for name, value in globals().items()
    if not name.startswith("_") and not inspect.ismodule(value)
]
