"""Utilities for recording training progress for dashboard consumption."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TrainingProgressTracker:
    """Persist coarse-grained training progress to a JSON file."""

    output_path: Path = field(
        default_factory=lambda: Path("reports") / "training" / "progress.json"
    )
    total_steps: int = 5

    def __post_init__(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._current_step = 0
        self._status = "idle"
        self._write(status="idle", stage="initialising", step=0)

    def start(self, stage: str = "initialising") -> None:
        self._current_step = 0
        self._status = "running"
        self._write(status="running", stage=stage, step=self._current_step)

    def advance(self, stage: str) -> None:
        if self._status != "running":
            self._status = "running"
        self._current_step += 1
        step = min(self._current_step, self.total_steps)
        self._write(status=self._status, stage=stage, step=step)

    def complete(self, runtime_seconds: float | None = None) -> None:
        self._status = "completed"
        payload: dict[str, Any] = {
            "status": self._status,
            "stage": "completed",
            "step": self.total_steps,
            "total_steps": self.total_steps,
            "updated_at": _now(),
        }
        if runtime_seconds is not None:
            payload["runtime_seconds"] = float(runtime_seconds)
        self._write_raw(payload)

    def fail(self, stage: str, error: Exception | str, runtime_seconds: float | None = None) -> None:
        self._status = "failed"
        payload: dict[str, Any] = {
            "status": self._status,
            "stage": stage,
            "step": min(self._current_step, self.total_steps),
            "total_steps": self.total_steps,
            "updated_at": _now(),
            "error": str(error),
        }
        if runtime_seconds is not None:
            payload["runtime_seconds"] = float(runtime_seconds)
        self._write_raw(payload)

    def _write(self, *, status: str, stage: str, step: int) -> None:
        payload = {
            "status": status,
            "stage": stage,
            "step": step,
            "total_steps": self.total_steps,
            "updated_at": _now(),
        }
        self._write_raw(payload)

    def _write_raw(self, payload: dict[str, Any]) -> None:
        try:
            self.output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        except Exception:
            logger.exception("Failed to persist training progress")


__all__ = ["TrainingProgressTracker"]

