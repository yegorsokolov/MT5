"""Commit and push log files to the main branch."""
from pathlib import Path
from git import Repo
from log_utils import setup_logging, log_exceptions

logger = setup_logging()

REPO_PATH = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_PATH / "logs"

@log_exceptions
def upload_logs() -> None:
    repo = Repo(REPO_PATH)
    repo.git.add(str(LOG_DIR))
    if not repo.is_dirty():
        logger.info("No new logs to commit")
        return
    repo.index.commit("Update logs")
    try:
        repo.remote().pull(rebase=True)
        repo.remote().push()
        logger.info("Pushed logs to remote")
    except Exception as e:
        logger.error("Failed to push logs: %s", e)


if __name__ == "__main__":
    upload_logs()
