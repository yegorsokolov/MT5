import time
from log_utils import setup_logging
from scripts.upload_logs import upload_logs

logger = setup_logging()

if __name__ == "__main__":
    while True:
        logger.info("Uploading logs...")
        try:
            upload_logs()
        except Exception as e:
            logger.error("Upload failed: %s", e)
        time.sleep(3600)
