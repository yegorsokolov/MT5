"""Periodically upload logs and checkpoints to the repository.

This helper is intended to be scheduled via cron or run as a background
process.  It also registers the shutdown hook so a final upload occurs when the
process exits.
"""

import time
import logging

from log_utils import setup_logging
from scripts.sync_artifacts import register_shutdown_hook, sync_artifacts

setup_logging()
logger = logging.getLogger(__name__)
register_shutdown_hook()


if __name__ == "__main__":
    while True:
        logger.info("Uploading artifacts...")
        try:
            sync_artifacts()
        except Exception as e:  # pragma: no cover - best effort logging
            logger.error("Upload failed: %s", e)
        time.sleep(3600)
