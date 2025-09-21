"""Pytest plugin for archiving logs after a test session.

This plugin mirrors log output and runtime artefacts into a deterministic
``logs/test_runs/<commit>/<timestamp>-<suffix>/`` folder so that auxiliary
scripts and CI pipelines can upload or inspect them later.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
import uuid

import pytest

ENV_DISABLE = "PYTEST_DISABLE_LOG_ARCHIVE"
ENV_EXTRA_DIRS = "PYTEST_LOG_ARCHIVE_EXTRA_DIRS"
LATEST_MARKER = "LATEST_RUN"


@dataclass
class _Source:
    """A directory or file that should be copied into the archive."""

    path: Path
    alias: Path
    ignore: tuple[Path, ...] = field(default_factory=tuple)


class _LogArchiveContext:
    """Holds configuration shared between pytest hooks."""

    def __init__(self, repo_root: Path, commit: str, run_id: str) -> None:
        self.repo_root = repo_root
        self.commit = commit
        self.run_id = run_id
        self.dest = repo_root / "logs" / "test_runs" / commit / run_id
        self.sources: list[_Source] = []

    def add_source(
        self,
        source: Path,
        alias: Path | None = None,
        ignore: Sequence[Path] | None = None,
    ) -> None:
        """Register a path to copy once the test session finishes."""

        if alias is None:
            try:
                alias = source.relative_to(self.repo_root)
            except ValueError:
                alias = Path(source.name or "external")
        if ignore is None:
            ignore = ()
        ignore_paths = tuple(Path(p) for p in ignore)
        self.sources.append(_Source(path=Path(source), alias=Path(alias), ignore=ignore_paths))

    def collect(self) -> Path | None:
        """Copy registered sources into the archive destination."""

        if not self.sources:
            return None
        created_any = False
        for source in self.sources:
            if not source.path.exists():
                continue
            if source.path.is_file():
                dest_path = self.dest / source.alias
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source.path, dest_path)
                created_any = True
                continue

            for path in source.path.rglob("*"):
                if path.is_dir():
                    continue
                if any(_is_relative_to(path, ignore_path) for ignore_path in source.ignore):
                    continue
                rel = path.relative_to(source.path)
                dest_path = self.dest / source.alias / rel
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest_path)
                created_any = True

        if not created_any:
            self.dest.mkdir(parents=True, exist_ok=True)

        marker = self.dest.parent / LATEST_MARKER
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(self.run_id, encoding="utf-8")

        return self.dest


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return start


def _current_commit(repo_root: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(repo_root)
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return output.decode().strip() or "unknown"


def _generate_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return f"{timestamp}-{suffix}"


def _split_env_paths(value: str | None) -> list[Path]:
    if not value:
        return []
    parts = [p for p in value.split(os.pathsep) if p]
    return [Path(p) for p in parts]


def _get_tmp_base(config: pytest.Config) -> Path | None:
    tmp_factory = getattr(config, "_tmp_path_factory", None)
    if tmp_factory is None:
        return None
    try:
        base = Path(str(tmp_factory.getbasetemp()))
    except Exception:  # pragma: no cover - defensive fallback
        return None
    return base


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("log-archive")
    group.addoption(
        "--no-log-archive",
        action="store_true",
        dest="log_archive_disable",
        help="Disable automatic collection of log artefacts for this session.",
    )
    parser.addini(
        "log_archive_extra_dirs",
        "Additional directories to snapshot into the test log archive.",
        type="pathlist",
        default=[],
    )


def pytest_configure(config: pytest.Config) -> None:
    if getattr(config, "workerinput", None) is not None:
        return
    if config.getoption("log_archive_disable"):
        return
    if os.environ.get(ENV_DISABLE):
        return

    repo_root = _find_repo_root(Path(__file__).resolve())
    commit = _current_commit(repo_root)
    run_id = _generate_run_id()
    context = _LogArchiveContext(repo_root=repo_root, commit=commit, run_id=run_id)

    logs_dir = repo_root / "logs"
    if logs_dir.exists():
        context.add_source(logs_dir, alias=Path("logs"), ignore=[logs_dir / "test_runs"])

    basetemp = _get_tmp_base(config)
    if basetemp and basetemp.exists():
        context.add_source(basetemp, alias=Path("pytest_tmp"))

    extra_dirs = list(config.getini("log_archive_extra_dirs"))
    extra_dirs.extend(_split_env_paths(os.environ.get(ENV_EXTRA_DIRS)))

    for raw_path in extra_dirs:
        extra_path = Path(raw_path)
        if not extra_path.is_absolute():
            extra_path = repo_root / extra_path
        if extra_path.exists():
            alias = Path("extras") / extra_path.name
            context.add_source(extra_path, alias=alias)

    config._log_archive_context = context  # type: ignore[attr-defined]


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    context = getattr(config, "_log_archive_context", None)
    if context is None:
        return
    if config.option.collectonly:
        return

    try:
        destination = context.collect()
    except Exception as exc:  # pragma: no cover - unexpected failures should not crash
        terminal = config.pluginmanager.get_plugin("terminalreporter")
        message = f"Log archive plugin failed: {exc}"
        if terminal is not None:
            terminal.write_line(message, red=True)
        else:
            print(message, file=sys.stderr)
        return

    if destination is None:
        return

    terminal = config.pluginmanager.get_plugin("terminalreporter")
    if terminal is not None:
        terminal.write_line(f"log archive created at {destination}")
