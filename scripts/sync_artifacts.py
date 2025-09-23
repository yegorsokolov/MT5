"""Commit and push logs and model checkpoints.

This utility mirrors the ``logs`` and ``checkpoints`` directories along with
selected configuration files into ``synced_artifacts/`` (or a custom
destination) before committing them to the repository. When the
``GITHUB_TOKEN`` environment variable is defined the token is injected into the
remote URL allowing unattended pushes from automation jobs.

Use :func:`register_shutdown_hook` to upload artifacts automatically when a
process exits.
"""

from __future__ import annotations

import atexit
import os
import shutil
import signal
from pathlib import Path
import logging

from git import Repo
from git.exc import GitCommandError
from mt5.log_utils import setup_logging, log_exceptions

_LOGGING_INITIALIZED = False
_HOOK_REGISTERED = False


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
DEFAULT_DESTINATION_ROOT = REPO_PATH / "synced_artifacts"

# Additional artefact directories and file types collected alongside logs.
DEFAULT_ARTIFACT_DIRS = [REPO_PATH / "analytics", REPO_PATH / "reports"]
DEFAULT_ARTIFACT_SUFFIXES = {
    ".csv",
    ".json",
    ".parquet",
    ".html",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".pdf",
    ".feather",
    ".xlsx",
}

ENV_ARTIFACT_DIRS = "SYNC_ARTIFACT_DIRS"
ENV_ARTIFACT_SUFFIXES = "SYNC_ARTIFACT_SUFFIXES"
ENV_ARTIFACT_DESTINATION = "SYNC_ARTIFACT_ROOT"
AUTO_SYNC_ENV = "AUTO_SYNC_ARTIFACTS"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


def _split_env_list(value: str | None) -> list[str]:
    """Return ``value`` split on common path separators."""

    if not value:
        return []
    normalised = value.replace(",", os.pathsep).replace(";", os.pathsep)
    return [item.strip() for item in normalised.split(os.pathsep) if item.strip()]


def _auto_sync_enabled() -> bool:
    """Return ``True`` when automatic artifact syncing is enabled."""

    flag = os.getenv(AUTO_SYNC_ENV)
    if flag is not None:
        return flag.strip().lower() in {"1", "true", "yes", "on"}
    return bool(GITHUB_TOKEN)


def _is_within_repo(path: Path) -> bool:
    """Return ``True`` when ``path`` is inside the repository root."""

    try:
        path.relative_to(REPO_PATH)
    except ValueError:
        return False
    return True


def _configured_suffixes() -> set[str]:
    """Return the set of file suffixes that should be synchronised."""

    suffixes = {s.lower() for s in DEFAULT_ARTIFACT_SUFFIXES}
    extra = _split_env_list(os.getenv(ENV_ARTIFACT_SUFFIXES))
    for suffix in extra:
        if not suffix:
            continue
        suffix = suffix if suffix.startswith(".") else f".{suffix}"
        suffixes.add(suffix.lower())
    return suffixes


def _configured_artifact_dirs() -> list[Path]:
    """Return additional directories requested via environment variables."""

    dirs: list[Path] = []
    for entry in _split_env_list(os.getenv(ENV_ARTIFACT_DIRS)):
        candidate = (REPO_PATH / entry).resolve()
        if not _is_within_repo(candidate):
            logger.warning("Skipping artifact directory outside repository: %s", candidate)
            continue
        dirs.append(candidate)
    return dirs


def _iter_additional_artifacts() -> list[Path]:
    """Collect analytics and report artefacts that should be committed."""

    suffixes = _configured_suffixes()
    directories = []
    seen_dirs: set[Path] = set()
    for directory in [*DEFAULT_ARTIFACT_DIRS, *_configured_artifact_dirs()]:
        if directory in seen_dirs:
            continue
        seen_dirs.add(directory)
        if directory.exists() and _is_within_repo(directory):
            directories.append(directory)
    artefacts: set[Path] = set()
    for directory in directories:
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in suffixes:
                artefacts.add(file_path)
    return sorted(artefacts)


def _gather_artifact_targets() -> list[Path]:
    """Return directories and files that should be staged for commit."""

    targets: list[Path] = []
    seen: set[Path] = set()
    for path in [LOG_DIR, CHECKPOINT_DIR, *CONFIG_FILES]:
        if path.exists() and path not in seen:
            targets.append(path)
            seen.add(path)
    for path in _iter_additional_artifacts():
        if path not in seen:
            targets.append(path)
            seen.add(path)
    return targets


def _resolve_destination_root() -> Path:
    """Return the directory under which artifacts are archived."""

    raw = os.getenv(ENV_ARTIFACT_DESTINATION)
    if not raw:
        return DEFAULT_DESTINATION_ROOT

    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (REPO_PATH / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not _is_within_repo(candidate):
        logger.warning(
            "Artifact destination %s is outside the repository; falling back to %s",
            candidate,
            DEFAULT_DESTINATION_ROOT.relative_to(REPO_PATH),
        )
        return DEFAULT_DESTINATION_ROOT

    if candidate == REPO_PATH:
        logger.warning(
            "Artifact destination cannot be the repository root; using %s",
            DEFAULT_DESTINATION_ROOT.relative_to(REPO_PATH),
        )
        return DEFAULT_DESTINATION_ROOT

    return candidate


def _prepare_destination_root(destination: Path) -> None:
    """Ensure the destination directory exists and is empty."""

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)


def _copy_to_destination(path: Path, destination_root: Path) -> Path | None:
    """Copy ``path`` into ``destination_root`` preserving its relative path."""

    try:
        rel = path.relative_to(REPO_PATH)
    except ValueError:
        logger.warning("Skipping artifact outside repository: %s", path)
        return None

    destination = destination_root / rel
    if destination_root in path.parents or path == destination_root:
        logger.debug("Skipping nested destination path %s", path)
        return None

    if path.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(path, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return destination


def _stage_destination_root(repo: Repo, destination: Path) -> None:
    """Stage the mirrored artifact tree for commit."""

    try:
        rel = destination.relative_to(REPO_PATH)
    except ValueError:
        logger.warning("Destination outside repository, not staging: %s", destination)
        return
    repo.git.add("--all", str(rel))


def _stage_path(repo: Repo, path: Path) -> None:
    """Stage ``path`` relative to the repository root."""

    try:
        rel = path.relative_to(REPO_PATH)
    except ValueError:
        logger.warning("Skipping path outside repository: %s", path)
        return
    repo.git.add(str(rel))


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
    """Commit and push logs, checkpoints, analytics and reports."""

    init_logging()
    repo = Repo(REPO_PATH)

    destination_root = _resolve_destination_root()

    targets = _gather_artifact_targets()
    if not targets:
        logger.info("No artifact paths found to stage")
        return

    base_targets = {LOG_DIR, CHECKPOINT_DIR, *CONFIG_FILES}
    extra_targets = [p for p in targets if p not in base_targets]
    try:
        destination_display = destination_root.relative_to(REPO_PATH)
    except ValueError:
        destination_display = destination_root
    logger.info(
        "Mirroring %d artifact paths into %s", len(targets), destination_display
    )
    if extra_targets:
        logger.info(
            "Including %d analytics/report artefacts", len(extra_targets)
        )

    _prepare_destination_root(destination_root)

    for path in targets:
        _copy_to_destination(path, destination_root)

    _stage_destination_root(repo, destination_root)

    for config_path in CONFIG_FILES:
        if config_path.exists():
            _stage_path(repo, config_path)

    if not repo.is_dirty():
        logger.info("No new artifacts or configs to commit")
        return

    repo.index.commit("Update synced logs, checkpoints and analytics")
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

    global _HOOK_REGISTERED
    if _HOOK_REGISTERED or not _auto_sync_enabled():
        return

    def _handler(*_: object) -> None:
        try:
            sync_artifacts()
        finally:
            raise SystemExit(0)

    atexit.register(sync_artifacts)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)
    logger.info("Registered automatic artifact sync on shutdown")
    _HOOK_REGISTERED = True


if __name__ == "__main__":
    sync_artifacts()
