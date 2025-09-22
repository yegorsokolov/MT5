"""Logging utilities for the trading bot."""

from __future__ import annotations

import csv
import json
import logging
from logging.handlers import RotatingFileHandler, SysLogHandler
from urllib.parse import urlparse
import socket
import io
import queue
import threading
from typing import NamedTuple

import requests
from functools import wraps
from pathlib import Path
from datetime import datetime, UTC

import pandas as pd
from mt5.metrics import ERROR_COUNT, TRADE_COUNT

try:  # optional during tests
    from core import state_sync
except Exception:  # pragma: no cover - optional dependency
    state_sync = None
from mt5.crypto_utils import _load_key, encrypt, decrypt

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"
TRADE_LOG = LOG_DIR / "trades.csv"
DECISION_LOG = LOG_DIR / "decisions.parquet.enc"
TRADE_HISTORY = LOG_DIR / "trade_history.parquet"
ORDER_ID_INDEX = LOG_DIR / "trade_history_order_ids.parquet"

# In-memory cache of order ids for quick duplicate detection
_order_id_cache: set[str] | None = None

# Standard column order for the trade CSV file.  New fields are appended to
# this list when first encountered to keep the header stable.
TRADE_COLUMNS = [
    "timestamp",
    "event",
    "symbol",
    "price",
    "qty",
    "order_id",
    "pnl",
    "exit_time",
    "model",
    "regime",
]


def _load_order_id_index() -> set[str]:
    """Load the cached set of order ids from disk if needed."""
    global _order_id_cache
    if _order_id_cache is None:
        if ORDER_ID_INDEX.exists():
            try:
                ids = pd.read_parquet(ORDER_ID_INDEX, columns=["order_id"])
                _order_id_cache = set(ids["order_id"].tolist())
            except Exception:
                _order_id_cache = set()
        else:
            _order_id_cache = set()
    return _order_id_cache


def _save_order_id_index() -> None:
    """Persist the cached order id set to disk."""
    if _order_id_cache is None:
        return
    df = pd.DataFrame({"order_id": list(_order_id_cache)})
    df.to_parquet(ORDER_ID_INDEX, engine="pyarrow")


# ---------------------------------------------------------------------------
# Asynchronous trade/decision logging
# ---------------------------------------------------------------------------

class _LogQueueItem(NamedTuple):
    """Item processed by the asynchronous logging worker."""

    kind: str
    payload: object
    ack: threading.Event | None


_SHUTDOWN_SENTINEL: object = object()


LOG_QUEUE: "queue.Queue[_LogQueueItem | object]" = queue.Queue()
_worker_thread: threading.Thread | None = None
_trade_handler: RotatingFileHandler | None = None
_decision_handler: RotatingFileHandler | None = None


def _get_trade_handler() -> RotatingFileHandler:
    """Return or create the rotating handler for the trade CSV."""
    global _trade_handler
    if _trade_handler is None:
        _trade_handler = RotatingFileHandler(
            TRADE_LOG, maxBytes=5 * 1024 * 1024, backupCount=5
        )
    return _trade_handler


def _get_decision_handler() -> RotatingFileHandler:
    """Return or create the rotating handler for the decision log."""
    global _decision_handler
    if _decision_handler is None:
        _decision_handler = RotatingFileHandler(
            DECISION_LOG, maxBytes=5 * 1024 * 1024, backupCount=5
        )
    return _decision_handler


def _log_decision_sync(df: pd.DataFrame, handler: RotatingFileHandler) -> None:
    """Write decisions DataFrame to encrypted parquet using ``handler``."""

    key = _load_key("DECISION_AES_KEY")
    if DECISION_LOG.exists():
        existing = read_decisions()
        df = pd.concat([existing, df], ignore_index=True)
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow")
    data = encrypt(buf.getvalue(), key)

    if DECISION_LOG.exists() and DECISION_LOG.stat().st_size >= handler.maxBytes:
        # ensure the handler stream is open for rollover
        if handler.stream is None:
            handler.stream = handler._open()
        handler.doRollover()
        handler.stream.close()
        handler.stream = None

    with open(DECISION_LOG, "wb") as f:
        f.write(data)

    if state_sync:
        state_sync.sync_decisions()


def _log_trade_sync(row: dict, t_handler: RotatingFileHandler, d_handler: RotatingFileHandler) -> None:
    """Synchronously write trade row using ``t_handler`` and record decision."""

    # Add timestamp and build row
    row = {"timestamp": datetime.now(UTC).isoformat(), **row}

    # Expand columns if needed
    new_cols = [c for c in row if c not in TRADE_COLUMNS]
    if new_cols:
        TRADE_COLUMNS.extend(new_cols)

    if t_handler.stream is None:
        t_handler.stream = t_handler._open()

    header_needed = t_handler.stream.tell() == 0
    if t_handler.stream.tell() >= t_handler.maxBytes:
        t_handler.doRollover()
        t_handler.stream = t_handler._open()
        header_needed = True

    writer = csv.DictWriter(t_handler.stream, fieldnames=TRADE_COLUMNS)
    if header_needed:
        writer.writeheader()
    writer.writerow({k: row.get(k, "") for k in TRADE_COLUMNS})
    t_handler.stream.flush()
    TRADE_COUNT.inc()

    _log_decision_sync(pd.DataFrame([row]), d_handler)


def _log_worker() -> None:
    """Background worker that processes queued log events."""

    t_handler = _get_trade_handler()
    d_handler = _get_decision_handler()
    while True:
        item = LOG_QUEUE.get()
        if item is _SHUTDOWN_SENTINEL:
            LOG_QUEUE.task_done()
            break
        assert isinstance(item, _LogQueueItem)
        try:
            if item.kind == "trade":
                _log_trade_sync(item.payload, t_handler, d_handler)
            elif item.kind == "decision":
                _log_decision_sync(item.payload, d_handler)
        finally:
            if item.ack:
                item.ack.set()
            LOG_QUEUE.task_done()

    t_handler.close()
    d_handler.close()


def _ensure_worker() -> None:
    """Start logging worker thread if not already running."""

    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(target=_log_worker, daemon=True)
        _worker_thread.start()


def shutdown_logging() -> None:
    """Flush and stop the logging worker."""

    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        LOG_QUEUE.put(_SHUTDOWN_SENTINEL)
        LOG_QUEUE.join()
        _worker_thread.join()
        _worker_thread = None


class JsonFormatter(logging.Formatter):
    """Simple JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        trace_id = getattr(record, "otelTraceID", None)
        span_id = getattr(record, "otelSpanID", None)
        if trace_id:
            log_record["trace_id"] = trace_id
            log_record["span_id"] = span_id
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


class ResilientHTTPHandler(logging.Handler):
    """HTTP log handler with simple retry buffer."""

    def __init__(self, url: str):
        super().__init__()
        self.url = url
        self.session = requests.Session()
        self.buffer: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - network
        msg = self.format(record)
        self.buffer.append(msg)
        self.flush()

    def flush(self) -> None:  # pragma: no cover - network
        while self.buffer:
            payload = self.buffer[0]
            try:
                self.session.post(
                    self.url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=5,
                )
                self.buffer.pop(0)
            except Exception:
                break


class ResilientSysLogHandler(logging.Handler):
    """Wrapper around ``SysLogHandler`` that reconnects on failure."""

    def __init__(self, address: tuple[str, int]):
        super().__init__()
        self.address = address
        self._connect()

    def _connect(self) -> None:  # pragma: no cover - system specific
        self.handler = SysLogHandler(address=self.address)

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - network
        try:
            self.handler.emit(record)
        except Exception:
            try:
                self.handler.close()
            except Exception:
                pass
            try:
                self._connect()
                self.handler.emit(record)
            except Exception:
                pass


def _is_reachable(url: str) -> bool:  # pragma: no cover - network
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"}:
        try:
            requests.head(url, timeout=5)
            return True
        except Exception:
            return False
    if parsed.scheme == "syslog":
        port = parsed.port or 514
        try:
            with socket.create_connection((parsed.hostname or "", port), timeout=5):
                return True
        except OSError:
            return False
    return False


def setup_logging() -> logging.Logger:
    """Configure and return the root logger."""
    logger = logging.getLogger()
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = JsonFormatter()

    fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    try:
        from utils import load_config

        cfg = load_config().model_dump()
    except Exception:
        cfg = {}

    url = (
        cfg.get("log_forward", {}).get("url")
        if isinstance(cfg.get("log_forward"), dict)
        else None
    )
    if url and _is_reachable(url):  # pragma: no branch - simple check
        parsed = urlparse(url)
        handler: logging.Handler | None = None
        if parsed.scheme in {"http", "https"}:
            handler = ResilientHTTPHandler(url)
        elif parsed.scheme == "syslog":
            address = (parsed.hostname or "localhost", parsed.port or 514)
            handler = ResilientSysLogHandler(address)
        if handler:
            handler.setFormatter(fmt)
            logger.addHandler(handler)

    try:  # pragma: no cover - systemd optional
        from systemd.journal import JournalHandler

        jh = JournalHandler(SYSLOG_IDENTIFIER="mt5bot")
        jh.setFormatter(fmt)
        logger.addHandler(jh)
    except Exception:
        pass

    return logger


def log_exceptions(func):
    """Decorator that logs entry, exit and exceptions for ``func``."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        log = logging.getLogger(func.__module__)
        log.info("Start %s", func.__name__)
        try:
            result = func(*args, **kwargs)
            log.info("End %s", func.__name__)
            return result
        except Exception:
            log.exception("Error in %s", func.__name__)
            ERROR_COUNT.inc()
            raise

    return wrapper


def _enqueue_log(kind: str, payload: object, flush: bool) -> None:
    """Submit a logging task to the worker, optionally waiting for completion."""

    _ensure_worker()
    ack = threading.Event() if flush else None
    LOG_QUEUE.put(_LogQueueItem(kind, payload, ack))
    if ack is not None:
        ack.wait()


def log_trade(event: str, *, flush: bool = False, **fields) -> None:
    """Queue a trade event for asynchronous logging.

    Parameters
    ----------
    event:
        Event type for the trade row.
    flush:
        If ``True`` the call blocks until the trade has been written to disk.
        Defaults to ``False`` for non-blocking behaviour.
    **fields:
        Additional trade metadata columns.
    """

    _enqueue_log("trade", {"event": event, **fields}, flush)


def log_trade_history(record: dict) -> None:
    """Append executed trade with features to the parquet trade history.

    A lightweight index of ``order_id`` values is used to avoid scanning the
    full history for duplicates.
    """

    order_id = record.get("order_id")
    if order_id is not None:
        index = _load_order_id_index()
        if order_id in index:
            return

    df = pd.DataFrame([record])
    if TRADE_HISTORY.exists():
        try:
            df.to_parquet(TRADE_HISTORY, engine="pyarrow", append=True)
        except Exception:
            df.to_parquet(TRADE_HISTORY, engine="pyarrow")
    else:
        df.to_parquet(TRADE_HISTORY, engine="pyarrow")

    if order_id is not None:
        index = _load_order_id_index()
        index.add(order_id)
        _save_order_id_index()


def log_decision(df: pd.DataFrame, *, flush: bool = False) -> None:
    """Queue decision rows for asynchronous logging.

    Parameters
    ----------
    df:
        DataFrame with decision entries to append to the encrypted parquet log.
    flush:
        If ``True`` the call waits until the rows are persisted.
    """

    _enqueue_log("decision", df, flush)


def read_decisions() -> pd.DataFrame:
    """Return decrypted decisions DataFrame."""
    if not DECISION_LOG.exists():
        return pd.DataFrame()
    key = _load_key("DECISION_AES_KEY")
    data = decrypt(DECISION_LOG.read_bytes(), key)
    return pd.read_parquet(io.BytesIO(data))


def log_predictions(df: pd.DataFrame) -> None:
    """Log model predictions to the decisions store.

    The provided DataFrame should contain at least a ``Timestamp`` column.
    ``Timestamp`` will be renamed to ``timestamp`` and an ``event`` column
    with value ``prediction`` will be added before appending to the parquet
    file.
    """

    if "Timestamp" in df.columns:
        df = df.rename(columns={"Timestamp": "timestamp"})
    df = df.copy()
    df["event"] = "prediction"
    log_decision(df)
