"""Utility functions for configuration loading."""

from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import os
import yaml
import mlflow
import log_utils
from pydantic import ValidationError, BaseModel
from filelock import FileLock
from config_models import AppConfig, ConfigError
from .secret_manager import SecretManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_secrets(val):
    """Recursively replace ``secret://`` references with actual values."""

    if isinstance(val, dict):
        return {k: _resolve_secrets(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_resolve_secrets(v) for v in val]
    if isinstance(val, str) and val.startswith("secret://"):
        return SecretManager().get_secret(val) or ""
    return val


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML configuration and validate using :class:`AppConfig`."""
    if path is None:
        cfg_path = os.getenv("CONFIG_FILE")
        if cfg_path:
            path = Path(cfg_path)
        else:
            path = PROJECT_ROOT / "config.yaml"
    else:
        path = Path(path)

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    data = _resolve_secrets(data)

    try:
        cfg = AppConfig(**data)
    except ValidationError as e:
        raise ConfigError(f"Invalid configuration: {e}") from e
    return cfg


def save_config(cfg: AppConfig | dict) -> None:
    """Persist configuration back to the YAML file."""
    cfg_path = os.getenv("CONFIG_FILE")
    if cfg_path:
        path = Path(cfg_path)
    else:
        path = PROJECT_ROOT / "config.yaml"
    data = cfg.model_dump() if isinstance(cfg, BaseModel) else cfg
    with open(path, "w") as f:
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

    cfg_path_env = os.getenv('CONFIG_FILE')
    if cfg_path_env:
        cfg_path = Path(cfg_path_env)
    else:
        cfg_path = PROJECT_ROOT / 'config.yaml'
    lock = FileLock(str(cfg_path) + '.lock')

    with lock:
        cfg = load_config().model_dump()
        old = cfg.get(key)
        if old == value:
            return

        cfg[key] = value
        save_config(cfg)

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
    "load_config",
    "save_config",
    "update_config",
    "mlflow_run",
]

from .environment import ensure_environment  # noqa: E402

# Perform an environment check on import so scripts automatically attempt
# to resolve missing dependencies and adjust configuration for low-spec VMs.
ensure_environment()
