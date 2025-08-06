from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Tuple
import joblib


def _checkpoint_dir(directory: str | None = None) -> Path:
    """Return the checkpoint directory, creating it if needed."""
    base = Path(directory or os.getenv("CHECKPOINT_DIR", "checkpoints"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def save_checkpoint(state: dict[str, Any], step: int, directory: str | None = None) -> Path:
    """Persist ``state`` at ``step`` to the checkpoint directory.

    Parameters
    ----------
    state: Mapping of training state to persist. Must contain only
        serialisable objects.
    step: Numerical step used to identify the checkpoint.
    directory: Optional override for the checkpoint directory.
    """
    ckpt_dir = _checkpoint_dir(directory)
    path = ckpt_dir / f"checkpoint_{step}.pkl"
    joblib.dump(state, path)
    return path


def load_latest_checkpoint(directory: str | None = None) -> Tuple[int, dict[str, Any]] | None:
    """Load the most recent checkpoint if it exists.

    Returns ``None`` if no checkpoints are present. Otherwise returns a
    tuple of ``(step, state)``.
    """
    ckpt_dir = Path(directory or os.getenv("CHECKPOINT_DIR", "checkpoints"))
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("checkpoint_*.pkl"))
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    step_str = latest.stem.split("_")[-1]
    try:
        step = int(step_str)
    except ValueError:
        step = 0
    state = joblib.load(latest)
    return step, state
