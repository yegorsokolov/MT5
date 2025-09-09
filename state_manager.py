from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Tuple
from datetime import datetime
import shutil
import joblib
from crypto_utils import _load_key, encrypt, decrypt

try:  # optional during tests
    from core import state_sync
except Exception:  # pragma: no cover - optional dependency
    state_sync = None


def _checkpoint_dir(directory: str | None = None) -> Path:
    """Return the checkpoint directory, creating it if needed.

    Checkpoints are organised under ``/data/checkpoints`` with a timestamped
    sub-directory for each run. The current timestamp is cached in the
    ``CHECKPOINT_TIMESTAMP`` environment variable so multiple checkpoints from
    the same process end up in the same folder. The active configuration file
    is also copied alongside the checkpoints for reproducibility.
    """

    base = Path(directory or os.getenv("CHECKPOINT_DIR", "/data/checkpoints"))
    ts = os.getenv("CHECKPOINT_TIMESTAMP")
    if ts is None:
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        os.environ["CHECKPOINT_TIMESTAMP"] = ts
    ckpt_dir = base / ts
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg = Path("config.yaml")
    if cfg.exists():
        shutil.copy2(cfg, ckpt_dir / "config.yaml")
    return ckpt_dir


def save_checkpoint(
    state: dict[str, Any], step: int, directory: str | None = None
) -> Path:
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


def load_latest_checkpoint(
    directory: str | None = None,
) -> Tuple[int, dict[str, Any]] | None:
    """Load the most recent checkpoint if it exists.

    Returns ``None`` if no checkpoints are present. Otherwise returns a
    tuple of ``(step, state)``.
    """
    if state_sync:
        state_sync.pull_checkpoints()
    base = Path(directory or os.getenv("CHECKPOINT_DIR", "/data/checkpoints"))
    if not base.exists():
        return None
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    latest_dir = max(subdirs, key=lambda p: p.stat().st_mtime)
    checkpoints = sorted(latest_dir.glob("checkpoint_*.pkl.enc"))
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


def _runtime_state_file(account_id: str | None = None) -> Path:
    """Return the runtime state file path for ``account_id``.

    If ``account_id`` is provided, the state is namespaced per MT5 account
    so that switching between accounts does not leak trading state.  When no
    ``account_id`` is given the legacy ``runtime_state.pkl`` location is
    used for backwards compatibility.
    """

    if account_id:
        return _STATE_DIR / f"runtime_state_{account_id}.pkl"
    return _STATE_FILE


def save_runtime_state(
    last_timestamp: str,
    open_positions: list[dict[str, Any]],
    model_versions: list[str],
    model_weights: dict[str, Any] | None = None,
    feature_scalers: dict[str, Any] | None = None,
    account_id: str | None = None,
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
    model_weights:
        Optional mapping of model identifiers to their serialized weights.
    feature_scalers:
        Optional mapping of feature scaler objects used in preprocessing.

    Parameters
    ----------
    account_id:
        Optional MT5 account identifier. When provided, the runtime state is
        stored in a file unique to that account. This allows users to switch
        accounts while retaining their model state and cutting access from the
        previous account.

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
    if model_weights is not None:
        state["model_weights"] = model_weights
    if feature_scalers is not None:
        state["feature_scalers"] = feature_scalers
    path = _runtime_state_file(account_id)
    joblib.dump(state, path)
    return path


def load_runtime_state(account_id: str | None = None) -> dict[str, Any] | None:
    """Load runtime trading state if present.

    Parameters
    ----------
    account_id:
        Optional MT5 account identifier. When provided, the state is loaded
        from the account-specific file. If no file exists for that account the
        legacy global state file is used as a fallback to allow migration of
        existing state when a user switches accounts.

    Returns
    -------
    dict or None
        Previously saved runtime state or ``None`` when no state has been
        saved yet.
    """

    path = _runtime_state_file(account_id)
    if not path.exists():
        if account_id:
            # Fallback to legacy state file when migrating to a new account
            path = _STATE_FILE
            if not path.exists():
                return None
        else:
            return None
    try:
        state: dict[str, Any] = joblib.load(path)
    except Exception:
        return None

    state.setdefault("last_timestamp", "")
    state.setdefault("open_positions", [])
    state.setdefault("model_versions", [])
    state.setdefault("model_weights", {})
    state.setdefault("feature_scalers", {})
    return state


# ---------------------------------------------------------------------------
# Replay state persistence
# ---------------------------------------------------------------------------
_REPLAY_TS_FILE = _STATE_DIR / "last_replay_timestamp.txt"


def save_replay_timestamp(ts: str) -> Path:
    """Persist the timestamp of the last reprocessed decision."""

    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    _REPLAY_TS_FILE.write_text(ts)
    return _REPLAY_TS_FILE


def load_replay_timestamp() -> str:
    """Return the last reprocessed decision timestamp if available."""

    if not _REPLAY_TS_FILE.exists():
        return ""
    return _REPLAY_TS_FILE.read_text().strip()


# ---------------------------------------------------------------------------
# User-specified risk limits
# ---------------------------------------------------------------------------
_RISK_FILE = _STATE_DIR / "user_risk.pkl"


def save_user_risk(
    daily_drawdown: float, total_drawdown: float, news_blackout_minutes: int
) -> Path:
    """Persist user-provided risk parameters."""

    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "daily_drawdown": daily_drawdown,
        "total_drawdown": total_drawdown,
        "news_blackout_minutes": news_blackout_minutes,
    }
    joblib.dump(data, _RISK_FILE)
    return _RISK_FILE


def load_user_risk() -> dict[str, Any] | None:
    """Load previously saved risk parameters if present."""

    if not _RISK_FILE.exists():
        return None
    try:
        data: dict[str, Any] = joblib.load(_RISK_FILE)
    except Exception:
        return None
    return data


# ---------------------------------------------------------------------------
# Strategy router state persistence
# ---------------------------------------------------------------------------
_ROUTER_FILE = _STATE_DIR / "router_state.pkl"


def save_router_state(
    champion: str | None,
    A: dict[str, Any],
    b: dict[str, Any],
    rewards: dict[str, float],
    counts: dict[str, int],
    history: list[tuple[Any, float, str]],
    total_plays: int,
) -> Path:
    """Persist the strategy router's state to disk.

    Parameters
    ----------
    champion: Current champion strategy name.
    A, b: LinUCB parameter matrices.
    rewards: Cumulative reward per strategy.
    counts: Number of plays per strategy.
    history: Recorded feature/reward pairs.
    total_plays: Total number of routing decisions made.
    """

    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "champion": champion,
        "A": A,
        "b": b,
        "rewards": rewards,
        "counts": counts,
        "history": history,
        "total_plays": total_plays,
    }
    joblib.dump(state, _ROUTER_FILE)
    return _ROUTER_FILE


def load_router_state() -> dict[str, Any] | None:
    """Load previously persisted strategy router state."""

    if not _ROUTER_FILE.exists():
        return None
    try:
        return joblib.load(_ROUTER_FILE)
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
