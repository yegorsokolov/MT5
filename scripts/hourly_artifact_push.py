"""Periodically mirror logs and checkpoints into the repository's ``synced_artifacts/`` archive.

This helper is intended to be scheduled via cron or run as a background
process.  It also registers the shutdown hook so a final upload occurs when the
process exits.
"""

import os
import time
import logging
from mt5.log_utils import setup_logging
from scripts.sync_artifacts import register_shutdown_hook, sync_artifacts

_LOGGING_INITIALIZED = False
DEFAULT_INTERVAL_SECONDS = 3600


def init_logging() -> logging.Logger:
    """Initialise structured logging for scheduled artifact uploads."""

    global _LOGGING_INITIALIZED
    if not _LOGGING_INITIALIZED:
        setup_logging()
        _LOGGING_INITIALIZED = True
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


def _resolve_interval() -> int:
    """Return the configured sync interval, defaulting to hourly."""

    raw = os.getenv("SYNC_INTERVAL_SECONDS")
    if raw is None:
        return DEFAULT_INTERVAL_SECONDS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid SYNC_INTERVAL_SECONDS=%s; using %d seconds", raw, DEFAULT_INTERVAL_SECONDS
        )
        return DEFAULT_INTERVAL_SECONDS
    if value < 60:
        logger.warning("SYNC_INTERVAL_SECONDS below minimum; clamping to 60 seconds")
        return 60
    return value


def main() -> None:
    init_logging()
    interval = _resolve_interval()
    logger.info("Artifact sync interval set to %d seconds", interval)
    register_shutdown_hook()
    while True:
        logger.info("Uploading artifacts...")
        try:
            sync_artifacts()
        except Exception as e:  # pragma: no cover - best effort logging
            logger.error("Upload failed: %s", e)
        time.sleep(interval)


if __name__ == "__main__":
    main()
