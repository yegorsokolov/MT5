"""Utility functions for configuration loading."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Any, Literal, overload
import os
import warnings
import yaml
from mt5 import log_utils
from pydantic import ValidationError, BaseModel
from filelock import FileLock
from mt5.config_models import AppConfig, ConfigError
from .secret_manager import SecretManager

try:
    import mlflow as _mlflow
except ImportError:
    class _MlflowShim:
        """Lightweight MLflow replacement when the dependency is absent."""

        _warned = False

        def _warn(self) -> None:
            if not self.__class__._warned:
                warnings.warn(
                    "MLflow is not installed; experiment logging is disabled.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.__class__._warned = True

        def set_tracking_uri(self, *args, **kwargs) -> None:
            self._warn()

        def set_experiment(self, *args, **kwargs) -> None:
            self._warn()

        @contextmanager
        def start_run(self, *args, **kwargs):
            self._warn()
            yield None

        def log_dict(self, *args, **kwargs) -> None:
            self._warn()

    mlflow = _MlflowShim()
else:
    mlflow = _mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_SECRET_PREFIX = "secret://"
_MASK = "***"


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _sanitize_value(resolved: Any, raw: Any | None, mask: str) -> Any:
    if isinstance(raw, str):
        if raw.startswith(_SECRET_PREFIX):
            return mask
        return resolved if resolved is not None else raw
    if isinstance(resolved, str) and resolved.startswith(_SECRET_PREFIX):
        return mask

    if isinstance(resolved, Mapping) or isinstance(raw, Mapping):
        resolved_map = resolved if isinstance(resolved, Mapping) else {}
        raw_map = raw if isinstance(raw, Mapping) else {}
        keys = set(resolved_map.keys()) | set(raw_map.keys())
        return {
            key: _sanitize_value(resolved_map.get(key), raw_map.get(key), mask)
            for key in keys
        }

    if _is_sequence(resolved) or _is_sequence(raw):
        resolved_seq = list(resolved) if _is_sequence(resolved) else []
        raw_seq = list(raw) if _is_sequence(raw) else []
        length = max(len(resolved_seq), len(raw_seq))
        return [
            _sanitize_value(
                resolved_seq[idx] if idx < len(resolved_seq) else None,
                raw_seq[idx] if idx < len(raw_seq) else None,
                mask,
            )
            for idx in range(length)
        ]

    return resolved if resolved is not None else raw


def sanitize_config(
    cfg: BaseModel | Mapping[str, Any] | Any,
    *,
    raw_cfg: Mapping[str, Any] | None = None,
    mask: str = _MASK,
) -> Any:
    """Return ``cfg`` with secret placeholders masked."""

    raw_data = raw_cfg
    if isinstance(cfg, BaseModel):
        resolved = cfg.model_dump()
        raw_data = raw_data or getattr(cfg, "_raw_config", None)
    elif isinstance(cfg, Mapping):
        resolved = dict(cfg)
        raw_data = raw_data or getattr(cfg, "_raw_config", None)
        resolved.pop("_raw_config", None)
    else:
        resolved = cfg

    return _sanitize_value(resolved, raw_data, mask)


def _config_path(path: str | Path | None = None) -> Path:
    """Return the configuration path, falling back to defaults."""

    if path is None:
        cfg_path = os.getenv("CONFIG_FILE")
        if cfg_path:
            return Path(cfg_path)
        return PROJECT_ROOT / "config.yaml"
    return Path(path)


def _resolve_secrets(val):
    """Recursively replace ``secret://`` references with actual values."""

    if isinstance(val, dict):
        return {k: _resolve_secrets(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_resolve_secrets(v) for v in val]
    if isinstance(val, str) and val.startswith("secret://"):
        return SecretManager().get_secret(val) or ""
    return val


def load_config_data(
    path: str | Path | None = None, *, resolve_secrets: bool = True
) -> dict[str, Any]:
    """Return the raw configuration mapping.

    Parameters
    ----------
    path:
        Optional path to load from. Defaults to the ``CONFIG_FILE`` environment
        variable or ``config.yaml`` under the project root.
    resolve_secrets:
        When ``True`` (default) secret placeholders are resolved via
        :class:`SecretManager`. When ``False`` the original placeholders are
        preserved.
    """

    cfg_path = _config_path(path)
    with open(cfg_path, "r") as f:
        data = yaml.safe_load(f) or {}
    return _resolve_secrets(data) if resolve_secrets else data


@overload
def load_config(
    path: str | Path | None = None, *, resolve_secrets: Literal[True] = True
) -> AppConfig:
    ...


@overload
def load_config(
    path: str | Path | None = None, *, resolve_secrets: Literal[False]
) -> dict[str, Any]:
    ...


def load_config(
    path: str | Path | None = None, *, resolve_secrets: bool = True
) -> AppConfig | dict[str, Any]:
    """Load the configuration optionally preserving raw placeholders."""

    data = load_config_data(path=path, resolve_secrets=resolve_secrets)
    raw_data: dict[str, Any] | None = None
    if resolve_secrets:
        raw_data = load_config_data(path=path, resolve_secrets=False)
    if not resolve_secrets:
        return data

    try:
        cfg = AppConfig(**data)
    except ValidationError as e:
        raise ConfigError(f"Invalid configuration: {e}") from e
    if raw_data is not None:
        object.__setattr__(cfg, "_raw_config", deepcopy(raw_data))
    return cfg


def save_config(cfg: AppConfig | dict, path: str | Path | None = None) -> None:
    """Persist configuration back to the YAML file."""

    cfg_path = _config_path(path)
    data = cfg.model_dump() if isinstance(cfg, BaseModel) else cfg
    with open(cfg_path, "w") as f:
        yaml.safe_dump(data, f)


_LOG_PATH = PROJECT_ROOT / 'logs' / 'config_changes.csv'
_LOG_PATH.parent.mkdir(exist_ok=True)

_RISK_KEYS = {
    'risk_per_trade',
    'max_daily_loss',
    'max_drawdown',
    'max_var',
    'max_stress_loss',
    'max_cvar',
    'rl_max_position',
}


def update_config(key: str, value, reason: str) -> None:
    """Update a single config value while logging the change.

    Certain risk parameters cannot be modified.
    """
    if key in _RISK_KEYS:
        raise ValueError(f'Modification of {key} is not allowed due to FTMO risk rules')

    cfg_path = _config_path()
    lock = FileLock(str(cfg_path) + '.lock')

    with lock:
        raw_cfg = load_config(resolve_secrets=False)
        if not isinstance(raw_cfg, dict):
            raw_cfg = raw_cfg.model_dump()
        old = raw_cfg.get(key)
        if old == value:
            return

        raw_cfg[key] = value

        resolved_cfg = _resolve_secrets(raw_cfg)
        try:
            AppConfig(**resolved_cfg)
        except ValidationError as e:
            raise ConfigError(f"Invalid configuration: {e}") from e

        save_config(raw_cfg)

        with open(_LOG_PATH, 'a') as f:
            f.write(f"{datetime.utcnow().isoformat()},{key},{old},{value},{reason}\n")


@contextmanager
def mlflow_run(experiment: str, cfg):
    """Start an MLflow run under ``logs/mlruns`` and log the given config."""
    logs_dir = getattr(log_utils, "LOG_DIR", PROJECT_ROOT / "logs")
    mlflow.set_tracking_uri(f"file:{logs_dir / 'mlruns'}")
    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        raw_cfg = None
        if isinstance(cfg, BaseModel):
            raw_cfg = getattr(cfg, "_raw_config", None)
        elif isinstance(cfg, Mapping):
            raw_cfg = getattr(cfg, "_raw_config", None)
        data = (
            cfg.model_dump()
            if isinstance(cfg, BaseModel)
            else (dict(cfg) if isinstance(cfg, Mapping) else cfg)
        )
        sanitized = sanitize_config(cfg, raw_cfg=raw_cfg)
        payload = (
            sanitized
            if isinstance(sanitized, Mapping)
            else raw_cfg
            if isinstance(raw_cfg, Mapping)
            else data
        )
        mlflow.log_dict(payload, "config.yaml")
        yield


__all__ = [
    "PROJECT_ROOT",
    "load_config_data",
    "load_config",
    "save_config",
    "update_config",
    "sanitize_config",
    "mlflow_run",
]

from .environment import ensure_environment  # noqa: E402

__all__.append("ensure_environment")
