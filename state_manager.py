from __future__ import annotations

from pathlib import Path
import os
import copy
import json
from typing import Any, Tuple, Callable
from datetime import datetime
import hashlib
import shutil
import logging
import threading
from collections.abc import Sequence
import joblib
try:
    from filelock import FileLock
except ImportError:  # pragma: no cover - lightweight fallback for tests
    class FileLock:  # type: ignore[misc]
        """Minimal file lock compatible with ``filelock.FileLock``."""

        def __init__(self, filename: str) -> None:
            self.filename = filename
            self._lock = threading.Lock()

        def acquire(self, *_: Any, **__: Any) -> None:
            self._lock.acquire()

        def release(self, *_: Any, **__: Any) -> None:
            self._lock.release()

        def __enter__(self) -> "FileLock":
            self.acquire()
            return self

        def __exit__(self, *_: Any) -> None:
            self.release()
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:  # pragma: no cover - watchdog is optional in tests
    class FileSystemEventHandler:  # type: ignore[misc]
        """Fallback handler used when watchdog is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            pass

    class Observer:  # type: ignore[misc]
        """No-op observer used in environments without watchdog."""

        def __init__(self) -> None:
            self.daemon = False

        def schedule(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def join(self, *_: Any) -> None:
            pass
from crypto_utils import _load_key, encrypt, decrypt
from utils import load_config
from config_models import AppConfig, ConfigError

logger = logging.getLogger(__name__)

try:  # optional during tests
    from core import state_sync
except Exception:  # pragma: no cover - optional dependency
    state_sync = None


class StateCorruptionError(Exception):
    """Raised when a checkpoint's integrity check fails."""


class _ConfigEventHandler(FileSystemEventHandler):
    """Reload configuration on file modifications."""

    def __init__(
        self,
        cfg: AppConfig,
        path: Path,
        callbacks: list[Callable[[AppConfig], None]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.path = path.resolve()
        self.callbacks = callbacks or []
        self._lock = threading.Lock()

    def on_modified(self, event):  # pragma: no cover - triggered by watchdog
        if Path(event.src_path).resolve() != self.path:
            return
        try:
            new_cfg = load_config(self.path)
        except ConfigError as exc:
            logger.warning("Invalid config update: %s", exc)
            return
        with self._lock:
            self.cfg.update_from(new_cfg)
            for cb in self.callbacks:
                try:
                    cb(self.cfg)
                except Exception:  # pragma: no cover - user callbacks
                    logger.exception("Config update callback failed")
            logger.info("Reloaded configuration from %s", self.path)


def watch_config(
    cfg: AppConfig,
    path: str | Path = "config.yaml",
    callbacks: list[Callable[[AppConfig], None]] | None = None,
) -> Observer:
    """Start watching ``path`` for changes and mutate ``cfg`` in-place."""

    cfg_path = Path(path).resolve()
    handler = _ConfigEventHandler(cfg, cfg_path, callbacks)
    observer = Observer()
    observer.schedule(handler, str(cfg_path.parent), recursive=False)
    observer.daemon = True
    observer.start()
    return observer


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
    encrypted = encrypt(data, key)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        f.write(encrypted)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    digest = hashlib.sha256(encrypted).hexdigest()
    hash_path = path.with_name(path.name + ".sha256")
    tmp_hash = hash_path.with_name(hash_path.name + ".tmp")
    with open(tmp_hash, "w") as f:
        f.write(digest)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_hash, hash_path)
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
    hash_path = latest.with_name(latest.name + ".sha256")
    if not hash_path.exists():
        raise StateCorruptionError("Missing checkpoint checksum")
    blob = latest.read_bytes()
    actual = hashlib.sha256(blob).hexdigest()
    expected = hash_path.read_text().strip()
    if actual != expected:
        raise StateCorruptionError("Checkpoint file corrupted")
    step_str = latest.stem.split("_")[-1]
    try:
        step = int(step_str)
    except ValueError:
        step = 0
    key = _load_key("CHECKPOINT_AES_KEY")
    data = decrypt(blob, key)
    state = joblib.loads(data)
    return step, state


# ---------------------------------------------------------------------------
# Runtime state persistence
# ---------------------------------------------------------------------------
_STATE_DIR = Path(os.getenv("MT5BOT_STATE_DIR", "/var/lib/mt5bot"))
_STATE_FILE = _STATE_DIR / "runtime_state.pkl"


def _ensure_state_dir() -> None:
    """Create the state directory with restrictive permissions."""
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    try:  # ensure only owner can access
        os.chmod(_STATE_DIR, 0o700)
    except PermissionError:  # pragma: no cover - best effort on readonly FS
        pass


def _lock(path: Path) -> FileLock:
    """Return a file lock for ``path``."""
    return FileLock(str(path) + ".lock")


def _runtime_state_file(account_id: str | None = None) -> Path:
    """Return the runtime state file path for ``account_id``.

    When ``account_id`` is provided the identifier is validated and the state
    file is namespaced per MT5 account so switching accounts does not leak
    trading state.
    """

    if account_id:
        if not str(account_id).isdigit():
            raise ValueError("Invalid account_id")
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

    _ensure_state_dir()
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
    with _lock(path):
        joblib.dump(state, path)
    return path


def load_runtime_state(account_id: str | None = None) -> dict[str, Any] | None:
    """Load runtime trading state if present.

    Parameters
    ----------
    account_id:
        Optional MT5 account identifier. When provided, the state is loaded
        from the account-specific file. No automatic fallback to the legacy
        global file is performed.

    Returns
    -------
    dict or None
        Previously saved runtime state or ``None`` when no state has been
        saved yet.
    """

    path = _runtime_state_file(account_id)
    if not path.exists():
        return None
    try:
        with _lock(path):
            state: dict[str, Any] = joblib.load(path)
    except Exception:
        return None

    state.setdefault("last_timestamp", "")
    state.setdefault("open_positions", [])
    state.setdefault("model_versions", [])
    state.setdefault("model_weights", {})
    state.setdefault("feature_scalers", {})
    return state


def legacy_runtime_state_exists() -> bool:
    """Return ``True`` if the pre-account namespaced state file exists."""

    return _STATE_FILE.exists()


def _parse_timestamp(ts: Any) -> datetime | None:
    """Return a ``datetime`` parsed from ``ts`` if possible."""

    if ts in (None, ""):
        return None
    try:
        text = str(ts)
    except Exception:  # pragma: no cover - defensive
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _select_timestamp(primary: Any, secondary: Any) -> str:
    """Return the most recent timestamp between ``primary`` and ``secondary``."""

    primary_dt = _parse_timestamp(primary)
    secondary_dt = _parse_timestamp(secondary)
    if primary_dt and secondary_dt:
        return str(primary if primary_dt >= secondary_dt else secondary)
    if primary_dt:
        return str(primary)
    if secondary_dt:
        return str(secondary)
    for candidate in (primary, secondary):
        if candidate not in (None, ""):
            return str(candidate)
    return ""


def _json_default(obj: Any) -> str:
    """Fallback serialiser used when deduplicating complex objects."""

    return repr(obj)


def _sequence_marker(item: Any) -> str:
    """Return a stable marker for ``item`` used during deduplication."""

    if isinstance(item, (str, int, float, bool)) or item is None:
        return f"scalar:{item!r}"
    try:
        return "json:" + json.dumps(item, sort_keys=True, default=_json_default)
    except TypeError:  # pragma: no cover - fallback when json fails
        return f"repr:{repr(item)}"


def _iter_sequence(value: Any) -> list[Any]:
    """Normalise ``value`` to a list for merging sequences."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def _merge_sequences(primary: Any, secondary: Any) -> list[Any]:
    """Merge two sequences while preserving order and removing duplicates."""

    merged: list[Any] = []
    seen: set[str] = set()
    for source in (_iter_sequence(primary), _iter_sequence(secondary)):
        for item in source:
            marker = _sequence_marker(item)
            if marker in seen:
                continue
            seen.add(marker)
            merged.append(copy.deepcopy(item))
    return merged


def _merge_mappings(primary: Any, secondary: Any) -> dict[Any, Any]:
    """Merge mapping-like objects with ``primary`` taking precedence."""

    merged: dict[Any, Any] = {}
    if isinstance(secondary, dict):
        for key, value in secondary.items():
            merged[key] = copy.deepcopy(value)
    if isinstance(primary, dict):
        for key, value in primary.items():
            merged[key] = copy.deepcopy(value)
    return merged


def _merge_runtime_states(
    new_state: dict[str, Any], legacy_state: dict[str, Any]
) -> dict[str, Any]:
    """Combine existing and legacy runtime states into a unified mapping."""

    if not isinstance(legacy_state, dict):
        raise ValueError("Legacy runtime state must be a mapping")
    if not isinstance(new_state, dict):
        new_state = {}
    merged: dict[str, Any] = {}
    for source in (legacy_state, new_state):
        for key, value in source.items():
            merged[key] = copy.deepcopy(value)
    merged["last_timestamp"] = _select_timestamp(
        new_state.get("last_timestamp"), legacy_state.get("last_timestamp")
    )
    merged["open_positions"] = _merge_sequences(
        new_state.get("open_positions"), legacy_state.get("open_positions")
    )
    merged["model_versions"] = _merge_sequences(
        new_state.get("model_versions"), legacy_state.get("model_versions")
    )
    merged["model_weights"] = _merge_mappings(
        new_state.get("model_weights"), legacy_state.get("model_weights")
    )
    merged["feature_scalers"] = _merge_mappings(
        new_state.get("feature_scalers"), legacy_state.get("feature_scalers")
    )
    merged.setdefault("last_timestamp", "")
    merged.setdefault("open_positions", [])
    merged.setdefault("model_versions", [])
    merged.setdefault("model_weights", {})
    merged.setdefault("feature_scalers", {})
    return merged


def migrate_runtime_state(account_id: str) -> Path:
    """Move or merge legacy runtime state into the account-specific file."""

    new_path = _runtime_state_file(account_id)
    legacy_path = _STATE_FILE
    if not legacy_path.exists():
        if new_path.exists():
            return new_path
        raise FileNotFoundError("Legacy runtime state not found")
    _ensure_state_dir()
    with _lock(legacy_path), _lock(new_path):
        if new_path.exists():
            try:
                legacy_state: dict[str, Any] = joblib.load(legacy_path)
            except Exception as exc:  # pragma: no cover - corrupted legacy state
                raise RuntimeError("Failed to load legacy runtime state") from exc
            try:
                new_state_raw = joblib.load(new_path)
                if not isinstance(new_state_raw, dict):
                    logger.warning(
                        "Existing runtime state %s is not a mapping; ignoring its contents",
                        new_path,
                    )
                    new_state: dict[str, Any] = {}
                else:
                    new_state = new_state_raw
            except Exception:
                logger.exception(
                    "Existing runtime state %s could not be loaded; replacing with legacy data",
                    new_path,
                )
                new_state = {}
            merged_state = _merge_runtime_states(new_state, legacy_state)
            tmp_path = new_path.with_name(new_path.name + ".tmp")
            try:
                joblib.dump(merged_state, tmp_path)
                os.replace(tmp_path, new_path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            try:
                legacy_path.unlink()
            except FileNotFoundError:  # pragma: no cover - already moved
                pass
            logger.info("Merged legacy runtime state into %s", new_path)
        else:
            shutil.move(legacy_path, new_path)
            logger.info("Migrated legacy runtime state to %s", new_path)
    return new_path


# ---------------------------------------------------------------------------
# Replay state persistence
# ---------------------------------------------------------------------------
_REPLAY_TS_FILE = _STATE_DIR / "last_replay_timestamp.txt"


def save_replay_timestamp(ts: str) -> Path:
    """Persist the timestamp of the last reprocessed decision."""

    _ensure_state_dir()
    with _lock(_REPLAY_TS_FILE):
        _REPLAY_TS_FILE.write_text(ts)
    return _REPLAY_TS_FILE


def load_replay_timestamp() -> str:
    """Return the last reprocessed decision timestamp if available."""

    if not _REPLAY_TS_FILE.exists():
        return ""
    with _lock(_REPLAY_TS_FILE):
        return _REPLAY_TS_FILE.read_text().strip()


# ---------------------------------------------------------------------------
# User-specified risk limits
# ---------------------------------------------------------------------------
_RISK_FILE = _STATE_DIR / "user_risk.pkl"


def save_user_risk(
    daily_drawdown: float,
    total_drawdown: float,
    news_blackout_minutes: int,
    allow_hedging: bool = False,
) -> Path:
    """Persist user-provided risk parameters."""

    _ensure_state_dir()
    data = {
        "daily_drawdown": daily_drawdown,
        "total_drawdown": total_drawdown,
        "news_blackout_minutes": news_blackout_minutes,
        "allow_hedging": allow_hedging,
    }
    with _lock(_RISK_FILE):
        joblib.dump(data, _RISK_FILE)
    return _RISK_FILE


def load_user_risk() -> dict[str, Any]:
    """Load risk parameters or return defaults.

    When no prior settings have been persisted the daily and total
    drawdown limits default to 4.9% and 9.8% of the initial capital
    respectively and the news blackout window defaults to ``0`` minutes.
    """

    if _RISK_FILE.exists():
        try:
            with _lock(_RISK_FILE):
                data: dict[str, Any] = joblib.load(_RISK_FILE)
                data.setdefault("allow_hedging", False)
                return data
        except Exception:
            pass
    ic = float(os.getenv("INITIAL_CAPITAL", "1.0"))
    return {
        "daily_drawdown": ic * 0.049,
        "total_drawdown": ic * 0.098,
        "news_blackout_minutes": 0,
        "allow_hedging": False,
    }


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
    """Persist the strategy router's state to disk."""

    _ensure_state_dir()
    state = {
        "champion": champion,
        "A": A,
        "b": b,
        "rewards": rewards,
        "counts": counts,
        "history": history,
        "total_plays": total_plays,
    }
    with _lock(_ROUTER_FILE):
        joblib.dump(state, _ROUTER_FILE)
    return _ROUTER_FILE


def load_router_state() -> dict[str, Any] | None:
    """Load previously persisted strategy router state."""

    if not _ROUTER_FILE.exists():
        return None
    try:
        with _lock(_ROUTER_FILE):
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
