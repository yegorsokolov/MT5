"""Utility functions for configuration loading."""

from pathlib import Path
from datetime import datetime
import yaml


def load_config() -> dict:
    """Load YAML configuration from the project root."""
    with open(Path(__file__).resolve().parent / 'config.yaml', 'r') as f:
        return yaml.safe_load(f)


def save_config(cfg: dict) -> None:
    """Persist configuration back to the YAML file."""
    path = Path(__file__).resolve().parent / 'config.yaml'
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f)


_LOG_PATH = Path(__file__).resolve().parent / 'logs' / 'config_changes.csv'
_LOG_PATH.parent.mkdir(exist_ok=True)

_RISK_KEYS = {
    'max_daily_loss',
    'max_drawdown',
    'max_var',
    'max_stress_loss',
}


def update_config(key: str, value, reason: str) -> None:
    """Update a single config value while logging the change.

    Certain risk parameters cannot be modified.
    """
    if key in _RISK_KEYS:
        raise ValueError(f'Modification of {key} is not allowed due to FTMO risk rules')

    cfg = load_config()
    old = cfg.get(key)
    if old == value:
        return

    cfg[key] = value
    save_config(cfg)

    with open(_LOG_PATH, 'a') as f:
        f.write(f"{datetime.utcnow().isoformat()},{key},{old},{value},{reason}\n")
