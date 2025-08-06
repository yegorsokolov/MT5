"""Logging utilities for the trading bot."""

from __future__ import annotations

import csv
import logging
from logging.handlers import RotatingFileHandler
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


def setup_logging() -> logging.Logger:
    """Configure and return the root logger."""
    logger = logging.getLogger()
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

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
