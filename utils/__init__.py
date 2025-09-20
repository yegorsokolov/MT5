"""Utility functions for configuration loading."""

from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Any, Literal, overload
import os
import yaml
import mlflow
import log_utils
from pydantic import ValidationError, BaseModel
from filelock import FileLock
from config_models import AppConfig, ConfigError
from .secret_manager import SecretManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    if not resolve_secrets:
        return data

    try:
        cfg = AppConfig(**data)
    except ValidationError as e:
        raise ConfigError(f"Invalid configuration: {e}") from e
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
        data = cfg.model_dump() if isinstance(cfg, BaseModel) else cfg
        mlflow.log_dict(data, "config.yaml")
        yield


__all__ = [
    "PROJECT_ROOT",
    "load_config_data",
    "load_config",
    "save_config",
    "update_config",
    "mlflow_run",
]

from .environment import ensure_environment  # noqa: E402

__all__.append("ensure_environment")
