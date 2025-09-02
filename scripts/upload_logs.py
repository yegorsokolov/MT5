"""Commit and push log files and configs to the main branch.

This script stages log files and selected configuration files,
committing them to the repository and pushing the update to the remote.
All run logs are copied into a ``Run logs`` directory so they remain
separate from local working files.  When the ``GITHUB_TOKEN`` environment
variable is present the token is injected into the remote URL so pushes can
occur from automation such as cron jobs.

The :func:`register_shutdown_hook` helper registers the upload to run when the
Python process exits which is useful for long running services.
"""

from __future__ import annotations

import atexit
import os
import signal
import shutil
from pathlib import Path
import logging

from git import Repo
from git.exc import GitCommandError

from log_utils import setup_logging, log_exceptions

setup_logging()
logger = logging.getLogger(__name__)

REPO_PATH = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_PATH / "logs"
RUN_LOG_DIR = REPO_PATH / "Run logs"
CONFIG_FILES = [REPO_PATH / "config.yaml"]

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


def _push_with_token(repo: Repo) -> None:
    """Push to the default remote using a GitHub token when provided."""

    origin = repo.remote()
    url = origin.url
    if GITHUB_TOKEN and url.startswith("https://") and "@" not in url:
        token_url = url.replace("https://", f"https://{GITHUB_TOKEN}@")
    else:
        token_url = None

    try:
        if token_url:
            repo.git.push(token_url, repo.active_branch.name)
        else:
            origin.push()
    except GitCommandError as exc:
        logger.error("Git push failed: %s", exc)
        raise


@log_exceptions
def upload_logs() -> None:
    """Commit and push logs and configuration files."""

    repo = Repo(REPO_PATH)

    if LOG_DIR.exists():
        shutil.rmtree(RUN_LOG_DIR, ignore_errors=True)
        for src in LOG_DIR.rglob("*"):
            if src.is_file():
                dest = RUN_LOG_DIR / src.relative_to(LOG_DIR)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)

    paths = [RUN_LOG_DIR, *CONFIG_FILES]
    for path in paths:
        if path.exists():
            repo.git.add(str(path))

    if not repo.is_dirty():
        logger.info("No new logs or configs to commit")
        return

    repo.index.commit("Update logs and configs")
    try:
        repo.remote().pull(rebase=True)
    except GitCommandError as exc:
        logger.warning("Failed to pull before push: %s", exc)
    try:
        _push_with_token(repo)
        logger.info("Pushed logs to remote")
    except GitCommandError as exc:
        logger.error("Failed to push logs: %s", exc)


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
