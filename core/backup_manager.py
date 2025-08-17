"""Backup management utilities.

This module handles archival of checkpoints and logs by compressing old files
and synchronising the resulting archives to a remote backend.  Supported
compression formats are ``xz`` and ``zstd`` (when the ``zstandard`` package is
available).  After synchronisation the integrity of the remote copy is
verified before the original file is deleted.  Local archives are pruned when
they exceed a retention window.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import lzma

try:  # optional dependency
    import zstandard as zstd  # type: ignore
except Exception:  # pragma: no cover - optional
    zstd = None  # type: ignore

logger = logging.getLogger(__name__)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str]) -> bool:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("backup command failed: %s", exc)
        return False


class BackupManager:
    """Compress and replicate old checkpoints and logs."""

    def __init__(
        self,
        checkpoint_dir: str | Path = "checkpoints",
        log_dir: str | Path = "logs",
        *,
        backend: str | None = None,
        retention_days: int | None = None,
        compress_after_days: int | None = None,
        compressor: str | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.backend = backend or os.getenv("BACKUP_BACKEND")
        self.retention_days = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))
        if retention_days is not None:
            self.retention_days = int(retention_days)
        self.compress_after = int(os.getenv("BACKUP_COMPRESS_AFTER", "1"))
        if compress_after_days is not None:
            self.compress_after = int(compress_after_days)
        self.compressor = (compressor or os.getenv("BACKUP_COMPRESSOR", "xz")).lower()
        self.status_file = Path("reports/backup_status.json")

    # ------------------------------------------------------------------
    def _compress(self, path: Path) -> Path:
        if self.compressor == "zstd" and zstd:
            out = path.with_suffix(path.suffix + ".zst")
            c = zstd.ZstdCompressor()
            with path.open("rb") as src, out.open("wb") as dst:
                c.copy_stream(src, dst)
        else:
            out = path.with_suffix(path.suffix + ".xz")
            with path.open("rb") as src, lzma.open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return out

    # ------------------------------------------------------------------
    def _remote_path(self, local: Path) -> str:
        assert self.backend  # for type checking
        return f"{self.backend.rstrip('/')}/{local.as_posix()}"

    # ------------------------------------------------------------------
    def _sync(self, local: Path) -> bool:
        if not self.backend:
            return True
        remote = self._remote_path(local)
        if self.backend.startswith("s3://"):
            return _run(["aws", "s3", "cp", str(local), remote])
        dst = Path(remote)
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, dst)
            return True
        except Exception:
            logger.exception("Failed copying %s to %s", local, dst)
            return False

    # ------------------------------------------------------------------
    def _verify_remote(self, local: Path) -> bool:
        if not self.backend:
            return True
        remote = self._remote_path(local)
        if self.backend.startswith("s3://"):
            tmp = local.with_suffix(local.suffix + ".tmp")
            if not _run(["aws", "s3", "cp", remote, str(tmp)]):
                return False
            ok = _md5(local) == _md5(tmp)
            tmp.unlink(missing_ok=True)
            return ok
        dst = Path(remote)
        if not dst.exists():
            return False
        return _md5(local) == _md5(dst)

    # ------------------------------------------------------------------
    def _compress_and_sync(self, directory: Path, pattern: str) -> None:
        if not directory.exists():
            return
        threshold = time.time() - self.compress_after * 24 * 3600
        for file in directory.glob(pattern):
            if file.suffix in {".xz", ".zst"}:
                continue
            if file.stat().st_mtime > threshold:
                continue
            try:
                archived = self._compress(file)
                ok = self._sync(archived) and self._verify_remote(archived)
                if ok:
                    file.unlink(missing_ok=True)
                    logger.info("Archived %s", file)
                else:
                    archived.unlink(missing_ok=True)
            except Exception:
                logger.exception("Failed archiving %s", file)

    # ------------------------------------------------------------------
    def _prune_archives(self, directory: Path) -> None:
        if not directory.exists():
            return
        cutoff = time.time() - self.retention_days * 24 * 3600
        for file in list(directory.glob("*.xz")) + list(directory.glob("*.zst")):
            if file.stat().st_mtime > cutoff:
                continue
            if not self._verify_remote(file):
                logger.warning("Skipping prune for %s; remote copy missing", file)
                continue
            try:
                file.unlink()
                logger.info("Pruned %s", file)
            except Exception:
                logger.exception("Failed pruning %s", file)

    # ------------------------------------------------------------------
    def _record_status(self, success: bool) -> None:
        status = {
            "last_run": datetime.utcnow().isoformat(),
            "last_success": bool(success),
        }
        try:
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            self.status_file.write_text(json.dumps(status))
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed writing backup status")
        try:
            from analytics.metrics_store import record_metric

            record_metric("backup_success", 1 if success else 0)
        except Exception:  # pragma: no cover - optional
            logger.warning("Failed recording backup metric")

    # ------------------------------------------------------------------
    def run(self) -> bool:
        """Execute the backup workflow."""
        success = True
        try:
            self._compress_and_sync(self.checkpoint_dir, "checkpoint_*.pkl.enc")
            self._compress_and_sync(self.log_dir, "*.log")
            self._compress_and_sync(self.log_dir, "*.parquet.enc")
            self._prune_archives(self.checkpoint_dir)
            self._prune_archives(self.log_dir)
        except Exception:
            logger.exception("Backup run failed")
            success = False
        self._record_status(success)
        return success


__all__ = ["BackupManager"]

