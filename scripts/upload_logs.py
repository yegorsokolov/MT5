"""Commit and push log files and configs to the main branch.

This script stages the contents of ``logs/`` and selected configuration files,
commits them to the repository and pushes the update to the remote.  When the
``GITHUB_TOKEN`` environment variable is present the token is injected into the
remote URL so pushes can occur from automation such as cron jobs.

The :func:`register_shutdown_hook` helper registers the upload to run when the
Python process exits which is useful for long running services.
"""

from __future__ import annotations

import atexit
import os
import signal
from pathlib import Path
import logging

from git import Repo

from log_utils import setup_logging, log_exceptions

setup_logging()
logger = logging.getLogger(__name__)

REPO_PATH = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_PATH / "logs"
CONFIG_FILES = [REPO_PATH / "config.yaml"]

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


def _push_with_token(repo: Repo) -> None:
    """Push to the default remote using a GitHub token when provided."""

    origin = repo.remote()
    url = origin.url
    if GITHUB_TOKEN and url.startswith("https://") and "@" not in url:
        token_url = url.replace("https://", f"https://{GITHUB_TOKEN}@")
        repo.git.push(token_url, repo.active_branch.name)
    else:
        origin.push()


@log_exceptions
def upload_logs() -> None:
    """Commit and push logs and configuration files."""

    repo = Repo(REPO_PATH)
    paths = [LOG_DIR, *CONFIG_FILES]
    for path in paths:
        if path.exists():
            repo.git.add(str(path))

    if not repo.is_dirty():
        logger.info("No new logs or configs to commit")
        return

    repo.index.commit("Update logs and configs")
    try:
        repo.remote().pull(rebase=True)
        _push_with_token(repo)
        logger.info("Pushed logs to remote")
    except Exception as e:
        logger.error("Failed to push logs: %s", e)


def register_shutdown_hook() -> None:
    """Trigger :func:`upload_logs` when the process terminates."""

    def _handler(*_: object) -> None:
        try:
            upload_logs()
        finally:
            raise SystemExit(0)

    atexit.register(upload_logs)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)


if __name__ == "__main__":
    upload_logs()
