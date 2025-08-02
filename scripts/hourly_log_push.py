"""Periodically upload logs and configs to the repository.

This helper is intended to be scheduled via cron or run as a background
process.  It also registers the shutdown hook so a final upload occurs when the
process exits.
"""

import time

from log_utils import setup_logging
from scripts.upload_logs import register_shutdown_hook, upload_logs

logger = setup_logging()
register_shutdown_hook()


if __name__ == "__main__":
    while True:
        logger.info("Uploading logs...")
        try:
            upload_logs()
        except Exception as e:
            logger.error("Upload failed: %s", e)
        time.sleep(3600)
