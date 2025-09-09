"""Utility functions for loading account configuration."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import yaml

# Path to account configuration YAML
CONFIG_FILE = Path(__file__).resolve().parents[2] / "config" / "account_config.yaml"


def _load_yaml(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_account_settings() -> Dict[str, str]:
    """Load account settings merging YAML and environment variables.

    Environment variables override values found in ``config/account_config.yaml``.
    The following variables are recognised:

    - ``ACCOUNT_ENVIRONMENT``
    - ``ACCOUNT_API_KEY``
    - ``ACCOUNT_API_SECRET``
    - ``ACCOUNT_ENDPOINT``
    """
    config = _load_yaml(CONFIG_FILE)
    return {
        "environment": os.getenv("ACCOUNT_ENVIRONMENT", config.get("environment", "demo")),
        "api_key": os.getenv("ACCOUNT_API_KEY", config.get("api_key")),
        "api_secret": os.getenv("ACCOUNT_API_SECRET", config.get("api_secret")),
        "endpoint": os.getenv("ACCOUNT_ENDPOINT", config.get("endpoint")),
    }
