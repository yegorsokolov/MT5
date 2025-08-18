from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Tuple
import joblib
from crypto_utils import _load_key, encrypt, decrypt
try:  # optional during tests
    from core import state_sync
except Exception:  # pragma: no cover - optional dependency
    state_sync = None


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
    path = ckpt_dir / f"checkpoint_{step}.pkl.enc"
    data = joblib.dumps(state)
    key = _load_key("CHECKPOINT_AES_KEY")
    path.write_bytes(encrypt(data, key))
    if state_sync:
        state_sync.sync_checkpoints()
    return path


def load_latest_checkpoint(directory: str | None = None) -> Tuple[int, dict[str, Any]] | None:
    """Load the most recent checkpoint if it exists.

    Returns ``None`` if no checkpoints are present. Otherwise returns a
    tuple of ``(step, state)``.
    """
    if state_sync:
        state_sync.pull_checkpoints()
    ckpt_dir = Path(directory or os.getenv("CHECKPOINT_DIR", "checkpoints"))
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("checkpoint_*.pkl.enc"))
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    step_str = latest.stem.split("_")[-1]
    try:
        step = int(step_str)
    except ValueError:
        step = 0
    key = _load_key("CHECKPOINT_AES_KEY")
    data = decrypt(latest.read_bytes(), key)
    state = joblib.loads(data)
    return step, state


# ---------------------------------------------------------------------------
# Runtime state persistence
# ---------------------------------------------------------------------------
_STATE_DIR = Path("/var/lib/mt5bot")
_STATE_FILE = _STATE_DIR / "runtime_state.pkl"


def save_runtime_state(
    last_timestamp: str,
    open_positions: list[dict[str, Any]],
    model_versions: list[str],
) -> Path:
    """Persist runtime trading state to the MT5 bot directory.

    Parameters
    ----------
    last_timestamp:
        ISO formatted timestamp of the last processed tick.
    open_positions:
        List of dictionaries describing currently open positions.
    model_versions:
        List of model version identifiers in use.

    Returns
    -------
    Path
        Location of the persisted state file.
    """

    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "last_timestamp": last_timestamp,
        "open_positions": open_positions,
        "model_versions": model_versions,
    }
    joblib.dump(state, _STATE_FILE)
    return _STATE_FILE


def load_runtime_state() -> dict[str, Any] | None:
    """Load runtime trading state if present.

    Returns ``None`` when no state has been saved yet."""

    if not _STATE_FILE.exists():
        return None
    try:
        return joblib.load(_STATE_FILE)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Service restart tracking
# ---------------------------------------------------------------------------
_RESTART_COUNTERS: dict[str, int] = {}


def increment_restart(service: str) -> int:
    """Increment and return the restart count for ``service``."""

    _RESTART_COUNTERS[service] = _RESTART_COUNTERS.get(service, 0) + 1
    return _RESTART_COUNTERS[service]


def get_restart_counters() -> dict[str, int]:
    """Return a copy of all service restart counters."""

    return dict(_RESTART_COUNTERS)
