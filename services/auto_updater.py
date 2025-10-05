"""Automatically pull Git updates and restart services when safe."""

from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable, Sequence

from mt5.config_models import AutoUpdateConfig
from utils import PROJECT_ROOT, load_config
from utils.market_hours import is_market_open

logger = logging.getLogger(__name__)


@dataclass
class UpdateState:
    """Metadata describing a pending remote commit."""

    remote_commit: str
    first_seen: datetime
    last_checked: datetime


class FileLock:
    """Simple advisory file lock backed by ``os.open`` semantics."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None

    def try_acquire(self) -> bool:
        """Return ``True`` if the lock was acquired successfully."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            return False
        try:
            os.write(self._fd, str(os.getpid()).encode())
        except Exception:
            self.release()
            raise
        return True

    def release(self) -> None:
        """Release the previously acquired lock if held."""

        if self._fd is None:
            return
        try:
            os.close(self._fd)
        finally:
            self._fd = None
            try:
                self.path.unlink(missing_ok=True)
            except Exception:
                pass


class AutoUpdater:
    """Encapsulates the logic for applying Git updates safely."""

    def __init__(
        self,
        *,
        settings: AutoUpdateConfig,
        repo_path: Path,
        symbols: Sequence[str],
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.repo_path = repo_path
        self.symbols = list(dict.fromkeys(symbols))
        self._now = now_fn or (lambda: datetime.now(UTC))

        if settings.state_file:
            self.state_path = Path(settings.state_file)
        else:
            self.state_path = repo_path / "logs" / "auto_update_state.json"
        if settings.lock_file:
            self.lock_path = Path(settings.lock_file)
        else:
            self.lock_path = repo_path / "logs" / "auto_update.lock"
        self._lock = FileLock(self.lock_path)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        *,
        repo_path: Path | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> "AutoUpdater":
        """Build an :class:`AutoUpdater` based on ``config.yaml``."""

        cfg = load_config()
        repo = Path(repo_path) if repo_path else PROJECT_ROOT
        settings = getattr(cfg, "auto_update", None)
        if settings is None:
            settings = AutoUpdateConfig()
        elif not isinstance(settings, AutoUpdateConfig):
            settings = AutoUpdateConfig.model_validate(settings)

        symbols: Sequence[str] = []
        strategy = getattr(cfg, "strategy", None)
        if strategy is not None and getattr(strategy, "symbols", None):
            symbols = list(strategy.symbols)
        else:
            raw = getattr(cfg, "symbols", None)
            if raw:
                symbols = list(raw)
        return cls(settings=settings, repo_path=repo, symbols=symbols, now_fn=now_fn)

    # ------------------------------------------------------------------
    def run(self, *, force: bool = False) -> bool:
        """Check for remote updates and apply them when appropriate.

        Returns ``True`` when new code was deployed.
        """

        if not self.settings.enabled:
            logger.info("Auto-update disabled in configuration")
            return False
        if not self._lock.try_acquire():
            logger.info("Another auto-update run is already in progress; skipping")
            return False
        try:
            return self._run_once(force=force)
        finally:
            self._lock.release()

    # ------------------------------------------------------------------
    def _run_once(self, *, force: bool) -> bool:
        remote_commit = self._fetch_remote_commit()
        local_commit = self._local_commit()
        if remote_commit == local_commit:
            logger.info("Repository already up to date (%s)", local_commit[:7])
            self._clear_state()
            return False

        state = self._load_state()
        now = self._now()
        if state is None or state.remote_commit != remote_commit:
            state = UpdateState(remote_commit=remote_commit, first_seen=now, last_checked=now)
        else:
            state.last_checked = now

        if self.settings.dry_run and not force:
            logger.info(
                "Dry run: update available %s -> %s",
                local_commit[:7],
                remote_commit[:7],
            )
            self._save_state(state)
            return False

        if not self._should_update(now, state, force=force):
            self._save_state(state)
            return False

        if self.settings.dry_run and not force:
            logger.info(
                "Dry run enabled: would update from %s to %s",
                local_commit[:7],
                remote_commit[:7],
            )
            self._save_state(state)
            return False

        logger.info(
            "Applying update from %s to %s", local_commit[:7], remote_commit[:7]
        )
        self._apply_update()
        self._clear_state()
        return True

    # ------------------------------------------------------------------
    def _fetch_remote_commit(self) -> str:
        self._git(["fetch", self.settings.remote, self.settings.branch])
        result = self._git(
            ["rev-parse", f"{self.settings.remote}/{self.settings.branch}"],
            capture_output=True,
        )
        return result.stdout.strip()

    # ------------------------------------------------------------------
    def _local_commit(self) -> str:
        result = self._git(["rev-parse", "HEAD"], capture_output=True)
        return result.stdout.strip()

    # ------------------------------------------------------------------
    def _should_update(self, now: datetime, state: UpdateState, *, force: bool) -> bool:
        if force or not self.settings.prefer_quiet_hours:
            return True
        total = len(self.symbols)
        if total == 0:
            return True
        open_symbols = self._open_symbols()
        fraction = len(open_symbols) / total if total else 0.0
        logger.debug(
            "Market state: %d/%d symbols active (%.0f%%)",
            len(open_symbols),
            total,
            fraction * 100,
        )
        if fraction <= self.settings.max_open_fraction:
            return True
        if self.settings.max_defer_minutes <= 0:
            return True
        age = now - state.first_seen
        if age >= timedelta(minutes=self.settings.max_defer_minutes):
            logger.warning(
                "Forcing update after %.1f minutes with markets busy", age.total_seconds() / 60
            )
            return True
        logger.info(
            "Deferring update; %.0f%% of tracked symbols still trading (%s)",
            fraction * 100,
            ", ".join(open_symbols) or "none",
        )
        return False

    # ------------------------------------------------------------------
    def _open_symbols(self) -> list[str]:
        open_symbols: list[str] = []
        for symbol in self.symbols:
            exchange = self.settings.exchanges.get(symbol) or self.settings.fallback_exchange
            if not exchange:
                continue
            normalised = exchange.strip().lower()
            if normalised in {"24/7", "always"}:
                open_symbols.append(symbol)
                continue
            try:
                if is_market_open(exchange):
                    open_symbols.append(symbol)
            except Exception:
                logger.exception("Failed to query market hours for %s (%s)", symbol, exchange)
                open_symbols.append(symbol)
        return open_symbols

    # ------------------------------------------------------------------
    def _apply_update(self) -> None:
        stashed = self._stash_changes()
        try:
            self._git(
                ["pull", "--ff-only", self.settings.remote, self.settings.branch],
                capture_output=False,
            )
        except Exception:
            if stashed:
                try:
                    self._git(["stash", "pop"], capture_output=False)
                except Exception:
                    logger.exception("Failed to restore stashed changes after pull error")
            raise
        if stashed:
            try:
                self._git(["stash", "pop"], capture_output=False)
            except subprocess.CalledProcessError:
                logger.exception("Failed to reapply stashed changes; manual resolution required")
                raise

        self._record_sync_event()
        self._ensure_protected_paths()
        self._restart_service()

    # ------------------------------------------------------------------
    def _stash_changes(self) -> bool:
        result = self._git(["status", "--porcelain"], capture_output=True)
        tracked = [line for line in result.stdout.splitlines() if line and not line.startswith("??")]
        if not tracked:
            return False
        logger.info("Stashing %d tracked change(s) before updating", len(tracked))
        self._git(["stash", "push", "--message", "auto-update"], capture_output=False)
        return True

    # ------------------------------------------------------------------
    def _restart_service(self) -> None:
        if self.settings.restart_command:
            cmd = list(self.settings.restart_command)
        elif self.settings.service_name:
            cmd = ["systemctl", "restart", self.settings.service_name]
        else:
            logger.info("No restart command configured; skipping service restart")
            return
        logger.info("Restarting service via: %s", " ".join(cmd))
        self._run(cmd)

    # ------------------------------------------------------------------
    def _ensure_protected_paths(self) -> None:
        for rel in self.settings.protected_paths:
            path = (self.repo_path / rel).resolve()
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception:
                logger.exception("Failed to ensure protected path %s", path)

    # ------------------------------------------------------------------
    def _record_sync_event(self) -> None:
        """Persist metadata about the most recent successful sync."""

        path = self.repo_path / "logs" / "last_sync.json"
        try:
            commit = (
                self._git(["rev-parse", "HEAD"], capture_output=True).stdout.strip()
            )
            commit_message = (
                self._git(["log", "-1", "--pretty=%B"], capture_output=True)
                .stdout.strip()
            )
            payload = {
                "synced_at": self._format_timestamp(self._now()),
                "commit": commit,
                "message": commit_message,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2))
        except Exception:
            logger.exception("Failed to record sync metadata")

    # ------------------------------------------------------------------
    def _load_state(self) -> UpdateState | None:
        if not self.state_path.exists():
            return None
        try:
            data = json.loads(self.state_path.read_text())
        except Exception:
            logger.exception("Failed to read state file %s", self.state_path)
            return None
        remote = data.get("remote_commit")
        first = self._parse_timestamp(data.get("first_seen"))
        last = self._parse_timestamp(data.get("last_checked")) or first
        if not remote or first is None:
            return None
        return UpdateState(remote_commit=remote, first_seen=first, last_checked=last)

    # ------------------------------------------------------------------
    def _save_state(self, state: UpdateState) -> None:
        payload = {
            "remote_commit": state.remote_commit,
            "first_seen": self._format_timestamp(state.first_seen),
            "last_checked": self._format_timestamp(state.last_checked),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(payload, indent=2))

    # ------------------------------------------------------------------
    def _clear_state(self) -> None:
        try:
            self.state_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed removing state file %s", self.state_path)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_timestamp(value: object) -> datetime | None:
        if not value:
            return None
        try:
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(float(value), tz=UTC)
            text = str(value)
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            return datetime.fromisoformat(text).astimezone(UTC)
        except Exception:
            return None

    # ------------------------------------------------------------------
    @staticmethod
    def _format_timestamp(value: datetime) -> str:
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")

    # ------------------------------------------------------------------
    def _git(
        self, args: Sequence[str], *, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        cmd = ["git", *args]
        logger.debug("Running %s", " ".join(cmd))
        return subprocess.run(
            cmd,
            cwd=self.repo_path,
            check=True,
            capture_output=capture_output,
            text=True,
        )

    # ------------------------------------------------------------------
    def _run(self, cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        logger.debug("Executing %s", " ".join(cmd))
        return subprocess.run(cmd, cwd=self.repo_path, check=True, text=True)


def _configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = RotatingFileHandler(log_dir / "auto_update.log", maxBytes=5_242_880, backupCount=3)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Automatically pull MT5 updates from Git")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Apply updates even when markets are open",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Override configuration to only log actions without pulling",
    )
    args = parser.parse_args(argv)

    updater = AutoUpdater.from_config()
    _configure_logging(updater.repo_path / "logs")

    if args.dry_run:
        updater.settings.dry_run = True

    try:
        changed = updater.run(force=args.force)
    except Exception:  # pragma: no cover - surfaced to systemd/journal
        logger.exception("Auto-update failed")
        raise
    return 0 if not changed else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
