"""Logging utilities for the trading bot."""

from __future__ import annotations

import builtins
import csv
import logging
from logging.handlers import RotatingFileHandler
from functools import wraps
from pathlib import Path
from datetime import datetime

from metrics import ERROR_COUNT, TRADE_COUNT

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"
TRADE_LOG = LOG_DIR / "trades.csv"


def setup_logging() -> logging.Logger:
    """Configure root logger and patch ``print`` to also log messages."""
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

    if not hasattr(builtins, "print_orig"):
        builtins.print_orig = builtins.print

        def patched_print(*args, **kwargs):
            message = " ".join(str(a) for a in args)
            logger.info(message)
            builtins.print_orig(*args, **kwargs)

        builtins.print = patched_print

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
    """Append a trade event to the CSV log."""
    header_needed = not TRADE_LOG.exists()
    with open(TRADE_LOG, "a", newline="") as f:
        cols = ["timestamp", "event"] + list(fields.keys())
        writer = csv.DictWriter(f, fieldnames=cols)
        if header_needed:
            writer.writeheader()
        row = {"timestamp": datetime.utcnow().isoformat(), "event": event}
        row.update(fields)
        writer.writerow(row)
    TRADE_COUNT.inc()
