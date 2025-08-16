"""Logging utilities for the trading bot."""

from __future__ import annotations

import csv
import json
import logging
from logging.handlers import RotatingFileHandler, SysLogHandler
from urllib.parse import urlparse
import os
import socket

import yaml
import requests
from functools import wraps
from pathlib import Path
from datetime import datetime, UTC

import pandas as pd

from metrics import ERROR_COUNT, TRADE_COUNT

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"
TRADE_LOG = LOG_DIR / "trades.csv"
DECISION_LOG = LOG_DIR / "decisions.parquet"
TRADE_HISTORY = LOG_DIR / "trade_history.parquet"


class JsonFormatter(logging.Formatter):
    """Simple JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
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

    cfg_path = os.getenv("CONFIG_FILE")
    if cfg_path:
        cfg_file = Path(cfg_path)
    else:
        cfg_file = Path(__file__).resolve().parent / "config.yaml"
    try:
        with open(cfg_file, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}

    url = cfg.get("log_forward", {}).get("url") if isinstance(cfg.get("log_forward"), dict) else None
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


def log_trade(event: str, **fields) -> None:
    """Append a trade event to the CSV log and decision store."""
    header_needed = not TRADE_LOG.exists()
    row = {"timestamp": datetime.now(UTC).isoformat(), "event": event}
    row.update(fields)
    with open(TRADE_LOG, "a", newline="") as f:
        cols = list(row.keys())
        writer = csv.DictWriter(f, fieldnames=cols)
        if header_needed:
            writer.writeheader()
        writer.writerow(row)
    TRADE_COUNT.inc()
    log_decision(pd.DataFrame([row]))


def log_trade_history(record: dict) -> None:
    """Append executed trade with features to the parquet trade history.

    If a trade with the same ``order_id`` is already present in the history the
    record is not duplicated.
    """

    df = pd.DataFrame([record])
    if TRADE_HISTORY.exists():
        try:
            if "order_id" in record:
                existing = pd.read_parquet(TRADE_HISTORY, columns=["order_id"])
                if record["order_id"] in set(existing["order_id"].tolist()):
                    return
            df.to_parquet(TRADE_HISTORY, engine="pyarrow", append=True)
        except Exception:
            df.to_parquet(TRADE_HISTORY, engine="pyarrow")
    else:
        df.to_parquet(TRADE_HISTORY, engine="pyarrow")


def log_decision(df: pd.DataFrame) -> None:
    """Append rows describing decisions to the parquet log."""
    if DECISION_LOG.exists():
        df.to_parquet(DECISION_LOG, engine="pyarrow", append=True)
    else:
        df.to_parquet(DECISION_LOG, engine="pyarrow")


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
