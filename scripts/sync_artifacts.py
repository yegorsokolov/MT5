"""Commit and push logs and model checkpoints.

This utility stages the ``logs`` and ``checkpoints`` directories along with
selected configuration files, committing them to the repository.  When the
``GITHUB_TOKEN`` environment variable is defined the token is injected into the
remote URL allowing unattended pushes from automation jobs.

Use :func:`register_shutdown_hook` to upload artifacts automatically when a
process exits.
"""

from __future__ import annotations

import atexit
import os
import signal
from pathlib import Path
import logging

from git import Repo
from git.exc import GitCommandError

from log_utils import setup_logging, log_exceptions

_LOGGING_INITIALIZED = False


def init_logging() -> logging.Logger:
    """Initialise structured logging for artifact synchronisation."""

    global _LOGGING_INITIALIZED
    if not _LOGGING_INITIALIZED:
        setup_logging()
        _LOGGING_INITIALIZED = True
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)

REPO_PATH = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_PATH / "logs"
CHECKPOINT_DIR = REPO_PATH / "checkpoints"
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
def sync_artifacts() -> None:
    """Commit and push logs, checkpoints and configuration files."""

    init_logging()
    repo = Repo(REPO_PATH)

    paths = [LOG_DIR, CHECKPOINT_DIR, *CONFIG_FILES]
    for path in paths:
        if path.exists():
            repo.git.add(str(path))

    if not repo.is_dirty():
        logger.info("No new artifacts or configs to commit")
        return

    repo.index.commit("Update logs and checkpoints")
    try:
        repo.remote().pull(rebase=True)
    except GitCommandError as exc:
        logger.warning("Failed to pull before push: %s", exc)
    try:
        _push_with_token(repo)
        logger.info("Pushed artifacts to remote")
    except GitCommandError as exc:
        logger.error("Failed to push artifacts: %s", exc)


def register_shutdown_hook() -> None:
    """Trigger :func:`sync_artifacts` when the process terminates."""

    def _handler(*_: object) -> None:
        try:
            sync_artifacts()
        finally:
            raise SystemExit(0)

    atexit.register(sync_artifacts)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)


if __name__ == "__main__":
    sync_artifacts()
