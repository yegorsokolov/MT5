"""Periodically upload logs and checkpoints to the repository.

This helper is intended to be scheduled via cron or run as a background
process.  It also registers the shutdown hook so a final upload occurs when the
process exits.
"""

import time
import logging

from log_utils import setup_logging
from scripts.sync_artifacts import register_shutdown_hook, sync_artifacts

_LOGGING_INITIALIZED = False


def init_logging() -> logging.Logger:
    """Initialise structured logging for scheduled artifact uploads."""

    global _LOGGING_INITIALIZED
    if not _LOGGING_INITIALIZED:
        setup_logging()
        _LOGGING_INITIALIZED = True
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


def main() -> None:
    init_logging()
    register_shutdown_hook()
    while True:
        logger.info("Uploading artifacts...")
        try:
            sync_artifacts()
        except Exception as e:  # pragma: no cover - best effort logging
            logger.error("Upload failed: %s", e)
        time.sleep(3600)


if __name__ == "__main__":
    main()
