"""Runtime state helpers used to resume multi-stage workflows.

The primary consumer is :mod:`mt5.pipeline_runner` which coordinates the
training → backtest → strategy → realtime pipeline.  Each stage persists a
succinct JSON snapshot describing whether resuming from that point is safe and
which artefacts must be present.  The :func:`python -m mt5` entry point queries
this information so restarts can continue from the last completed stage when
possible.

The module is intentionally lightweight: it only depends on the standard
library and the minimal ``FileLock`` implementation exposed by
``mt5.state_manager`` to avoid importing heavy optional dependencies during
tests.  All paths stored in the state file are recorded relative to the
repository root so the metadata remains stable even when the project is
checked out in a different location.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # Prefer the shared FileLock implementation used by state_manager.
    from mt5.state_manager import FileLock  # type: ignore
except Exception:  # pragma: no cover - fallback used in minimal environments.
    from threading import Lock

    class FileLock:  # type: ignore
        def __init__(self, *_: Any, **__: Any) -> None:
            self._lock = Lock()

        def __enter__(self) -> "FileLock":
            self._lock.acquire()
            return self

        def __exit__(self, *_: Any) -> None:
            self._lock.release()


PIPELINE_STAGES: tuple[str, ...] = (
    "preflight",
    "training",
    "backtest",
    "strategy",
    "optimise",
    "realtime",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_repo_root(start: Path | None = None) -> Path:
    candidate = start or Path(__file__).resolve()
    for base in (candidate, *candidate.parents):
        if (base / ".git").exists():
            return base
    return candidate.parent


def _normalise_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _materialise_path(path: str, repo_root: Path) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    return repo_root / raw


@dataclass
class StageRecord:
    """Metadata describing a single pipeline stage."""

    name: str
    status: str = "pending"
    resume: bool = True
    artifacts: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "resume": bool(self.resume),
            "artifacts": list(self.artifacts),
            "metrics": dict(self.metrics),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageRecord":
        return cls(
            name=str(payload.get("name")),
            status=str(payload.get("status", "pending")),
            resume=bool(payload.get("resume", True)),
            artifacts=[str(a) for a in payload.get("artifacts", [])],
            metrics=dict(payload.get("metrics", {})),
            started_at=payload.get("started_at"),
            completed_at=payload.get("completed_at"),
            error=payload.get("error"),
        )


@dataclass
class PipelineState:
    """Persist coarse pipeline progress for automatic resumption."""

    path: Path
    stages: Sequence[str] = PIPELINE_STAGES
    repo_root: Path | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.repo_root = (self.repo_root or _find_repo_root(self.path)).resolve()
        self._lock = FileLock(str(self.path) + ".lock")
        self._data = self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "status": "idle",
                "updated_at": _now(),
                "stages": [StageRecord(name=s).to_dict() for s in self.stages],
            }
        try:
            payload = json.loads(self.path.read_text())
        except Exception:
            payload = {}
        stages = payload.get("stages") or [StageRecord(name=s).to_dict() for s in self.stages]
        payload["stages"] = stages
        return payload

    def _save(self) -> None:
        self._data["updated_at"] = _now()
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, sort_keys=True)

    def _stage_entry(self, name: str) -> StageRecord:
        for raw in self._data.get("stages", []):
            if raw.get("name") == name:
                return StageRecord.from_dict(raw)
        record = StageRecord(name=name)
        self._data.setdefault("stages", []).append(record.to_dict())
        return record

    def _update_stage(self, record: StageRecord) -> None:
        stages = self._data.setdefault("stages", [])
        for idx, raw in enumerate(stages):
            if raw.get("name") == record.name:
                stages[idx] = record.to_dict()
                break
        else:
            stages.append(record.to_dict())

    def _artifacts_exist(self, record: StageRecord) -> bool:
        if not record.artifacts:
            return True
        return all(_materialise_path(path, self.repo_root).exists() for path in record.artifacts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear runtime metadata so the next run starts from scratch."""

        with self._lock:
            self._data = {
                "status": "idle",
                "updated_at": _now(),
                "stages": [StageRecord(name=s).to_dict() for s in self.stages],
            }
            self._save()

    def begin_run(self, *, mode: str = "pipeline", args: Mapping[str, Any] | None = None, resume_from: str | None = None) -> None:
        with self._lock:
            self._data["status"] = "running"
            self._data["mode"] = mode
            self._data["started_at"] = _now()
            self._data["args"] = dict(args or {})
            self._data["resume_from"] = resume_from
            self._save()

    def mark_stage_started(self, stage: str) -> None:
        with self._lock:
            record = self._stage_entry(stage)
            record.status = "running"
            record.started_at = _now()
            record.error = None
            self._update_stage(record)
            self._save()

    def mark_stage_complete(
        self,
        stage: str,
        *,
        artifacts: Iterable[os.PathLike[str] | str] | None = None,
        metrics: Mapping[str, Any] | None = None,
        resume: bool | None = None,
    ) -> None:
        artifacts = list(artifacts or [])
        with self._lock:
            record = self._stage_entry(stage)
            record.status = "completed"
            record.completed_at = _now()
            record.metrics = dict(metrics or {})
            record.artifacts = [
                _normalise_path(Path(a), self.repo_root) for a in artifacts if str(a)
            ]
            if resume is not None:
                record.resume = bool(resume)
            elif artifacts:
                record.resume = True
            else:
                # Without artefacts we err on the cautious side: force rerun
                record.resume = False
            self._update_stage(record)
            self._save()

    def mark_stage_failed(self, stage: str, error: Exception | str) -> None:
        with self._lock:
            record = self._stage_entry(stage)
            record.status = "failed"
            record.error = str(error)
            record.completed_at = _now()
            self._update_stage(record)
            self._data["status"] = "failed"
            self._data["error"] = str(error)
            self._save()

    def mark_run_completed(self) -> None:
        with self._lock:
            self._data["status"] = "completed"
            self._data["completed_at"] = _now()
            self._save()

    def mark_run_failed(self, error: Exception | str) -> None:
        with self._lock:
            self._data["status"] = "failed"
            self._data["error"] = str(error)
            self._data["completed_at"] = _now()
            self._save()

    def should_resume(self) -> bool:
        """Return ``True`` when an unfinished run can be resumed safely."""

        status = self._data.get("status")
        if status not in {"running", "failed"}:
            return False
        for stage in self.stages:
            record = self._stage_entry(stage)
            if record.status != "completed":
                return True
            if not record.resume or not self._artifacts_exist(record):
                return False
        return False

    def resume_stage(self) -> str | None:
        """Return the first stage that still needs to run."""

        for stage in self.stages:
            record = self._stage_entry(stage)
            if record.status != "completed":
                return stage
            if not record.resume:
                # Stage completed but not safe to resume – rerun from here.
                return stage
            if not self._artifacts_exist(record):
                # Required artefacts missing – rerun from here.
                return stage
        return None

    def stage_resume_ready(self, stage: str) -> bool:
        record = self._stage_entry(stage)
        return (
            record.status == "completed"
            and record.resume
            and self._artifacts_exist(record)
        )


__all__ = ["PipelineState", "StageRecord", "PIPELINE_STAGES"]

