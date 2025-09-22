"""Utilities for recording training run artefacts in a Codex-friendly format."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

__all__ = ["RunHistoryRecorder"]


def _now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    """Format ``dt`` as an ISO8601 string."""

    return dt.astimezone(timezone.utc).isoformat()


def _find_repo_root(start: Path) -> Path:
    """Return the repository root based on ``start``."""

    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return start


def _current_commit(repo_root: Path) -> str:
    """Return the short git commit hash if available."""

    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(repo_root)
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return output.decode().strip() or "unknown"


def _normalise(value: Any) -> Any:
    """Best-effort conversion of ``value`` to a JSON-serialisable structure."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(k): _normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalise(v) for v in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _normalise(model_dump())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return _normalise(vars(value))
    return str(value)


def _json_default(value: Any) -> str:
    """Fallback serialiser used by :mod:`json`."""

    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


@dataclass
class _ArtifactSpec:
    """Configuration for copying run artefacts into the history folder."""

    source: Path
    destination: Path
    copy: bool = True
    optional: bool = True
    max_bytes: int | None = 1_000_000


@dataclass
class RunHistoryRecorder:
    """Persist structured metadata for training or evaluation runs."""

    component: str
    config: Mapping[str, Any] | None = None
    tags: Mapping[str, Any] | None = None
    extra: Mapping[str, Any] | None = None
    run_id: str | None = None
    repo_root: Path | None = None

    _metrics: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _context: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _artifacts: list[_ArtifactSpec] = field(default_factory=list, init=False, repr=False)
    _notes: list[str] = field(default_factory=list, init=False, repr=False)
    _errors: list[str] = field(default_factory=list, init=False, repr=False)
    _status: str = field(default="pending", init=False, repr=False)
    _started_at: datetime | None = field(default=None, init=False, repr=False)
    _ended_at: datetime | None = field(default=None, init=False, repr=False)
    _result: Any = field(default=None, init=False, repr=False)
    _finished: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.repo_root is None:
            self.repo_root = _find_repo_root(Path(__file__).resolve())
        else:
            self.repo_root = Path(self.repo_root)
        self.repo_root = self.repo_root.resolve()
        self.history_root = self.repo_root / "reports" / "run_history"
        self.history_root.mkdir(parents=True, exist_ok=True)
        self.run_id = self.run_id or self._generate_run_id()
        self.run_dir = self.history_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.commit = _current_commit(self.repo_root)
        self.tags = dict(self.tags or {})
        self.extra = dict(self.extra or {})
        if self.config is not None:
            self.config = _normalise(self.config)

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    def _generate_run_id(self) -> str:
        timestamp = _now().strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}-{uuid.uuid4().hex[:8]}"

    def start(self) -> "RunHistoryRecorder":
        """Mark the run as started."""

        if self._started_at is None:
            self._started_at = _now()
            self._status = "running"
        return self

    def finish(
        self,
        *,
        status: str | None = None,
        error: Exception | str | None = None,
        result: Any = None,
    ) -> None:
        """Persist the run metadata to disk."""

        if self._finished:
            return

        if self._started_at is None:
            self.start()

        if result is not None:
            self._result = result

        self._ended_at = _now()
        resolved_status = status or ("failed" if error else "completed")
        self._status = resolved_status

        if error is not None and not self._errors:
            self._errors.append(str(error))

        artifacts = self._materialise_artifacts()
        record = {
            "component": self.component,
            "run_id": self.run_id,
            "git_commit": self.commit,
            "status": self._status,
            "started_at": _iso(self._started_at),
            "ended_at": _iso(self._ended_at),
            "duration_seconds": (self._ended_at - self._started_at).total_seconds(),
            "config": self.config,
            "metrics": _normalise(self._metrics),
            "tags": _normalise(self.tags),
            "context": _normalise({**self.extra, **self._context}),
            "notes": list(self._notes),
            "errors": list(self._errors),
            "artifacts": artifacts,
            "result": _normalise(self._result),
        }

        record_path = self.run_dir / "run.json"
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=_json_default))

        self._update_index(record_path, record)
        self._finished = True

    def __enter__(self) -> "RunHistoryRecorder":  # pragma: no cover - convenience
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - convenience
        status = "failed" if exc else "completed"
        if exc is not None:
            self.add_error(str(exc))
        self.finish(status=status, error=exc)
        return False

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def set_metrics(self, metrics: Mapping[str, Any]) -> None:
        """Merge ``metrics`` into the run summary."""

        for key, value in metrics.items():
            self._metrics[str(key)] = _normalise(value)

    def update_context(self, **fields: Any) -> None:
        """Attach arbitrary serialisable fields to the run context."""

        for key, value in fields.items():
            self._context[str(key)] = _normalise(value)

    def add_artifact(
        self,
        path: os.PathLike[str] | str,
        *,
        dest_name: str | None = None,
        copy: bool = True,
        optional: bool = True,
        max_bytes: int | None = 1_000_000,
    ) -> None:
        """Register a file to capture inside the run directory."""

        source = Path(path)
        if not source.is_absolute():
            source = (self.repo_root / source).resolve()
        dest = Path(dest_name) if dest_name else Path(source.name)
        if dest.is_absolute() or any(part == ".." for part in dest.parts):
            raise ValueError("dest_name must be a relative path within the run directory")
        self._artifacts.append(
            _ArtifactSpec(
                source=source,
                destination=dest,
                copy=copy,
                optional=optional,
                max_bytes=max_bytes,
            )
        )

    def add_note(self, message: str) -> None:
        """Record a free-form note for the run."""

        self._notes.append(message)

    def add_error(self, message: str) -> None:
        """Record an error message associated with the run."""

        self._errors.append(message)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.repo_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _copy_file(self, source: Path, destination: Path, max_bytes: int | None) -> tuple[bool, int, int]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        total_size = source.stat().st_size
        truncated = False
        if max_bytes is not None and max_bytes > 0 and total_size > max_bytes:
            truncated = True
            with source.open("rb") as src, destination.open("wb") as dst:
                src.seek(-max_bytes, os.SEEK_END)
                snippet = src.read(max_bytes)
                header = (
                    f"... truncated to last {max_bytes} bytes of {total_size} bytes ...\n"
                ).encode("utf-8")
                dst.write(header)
                dst.write(snippet)
            written = destination.stat().st_size
        else:
            shutil.copy2(source, destination)
            written = destination.stat().st_size
        return truncated, total_size, written

    def _materialise_artifacts(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for spec in self._artifacts:
            entry = {
                "source": self._relative(spec.source),
                "requested_destination": spec.destination.as_posix(),
                "copied": False,
                "exists": spec.source.exists(),
                "destination": None,
                "truncated": False,
            }
            if not spec.source.exists():
                if not spec.optional:
                    self._errors.append(f"Missing artifact: {spec.source}")
                entries.append(entry)
                continue

            if not spec.copy:
                entries.append(entry)
                continue

            if spec.source.is_dir():
                # Avoid copying arbitrarily large directories; record listing instead.
                entry["note"] = "directory copy not supported; recorded listing"
                listing = sorted(p.as_posix() for p in spec.source.iterdir())
                entry["listing"] = listing
                entries.append(entry)
                continue

            dest_path = self.run_dir / spec.destination
            truncated, total_size, written = self._copy_file(spec.source, dest_path, spec.max_bytes)
            entry.update(
                {
                    "copied": True,
                    "destination": self._relative(dest_path),
                    "truncated": truncated,
                    "original_size": total_size,
                    "written_size": written,
                }
            )
            entries.append(entry)
        return entries

    def _update_index(self, record_path: Path, record: Mapping[str, Any]) -> None:
        index_path = self.history_root / "index.json"
        latest_path = self.history_root / "latest.json"
        summary = {
            "run_id": self.run_id,
            "component": self.component,
            "status": self._status,
            "started_at": record["started_at"],
            "ended_at": record["ended_at"],
            "duration_seconds": record["duration_seconds"],
            "record": self._relative(record_path),
            "result": record.get("result"),
            "git_commit": self.commit,
        }
        if index_path.exists():
            data = json.loads(index_path.read_text())
            runs = data.get("runs", [])
        else:
            runs = []
        runs.insert(0, summary)
        payload = {
            "generated_at": _iso(_now()),
            "runs": runs,
        }
        index_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        latest_path.write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "record": summary["record"],
                    "generated_at": payload["generated_at"],
                },
                indent=2,
                sort_keys=True,
            )
        )
