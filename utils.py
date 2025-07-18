"""Utility functions for configuration loading."""

from pathlib import Path
import yaml


def load_config() -> dict:
    """Load YAML configuration from the project root."""
    with open(Path(__file__).resolve().parent / 'config.yaml', 'r') as f:
        return yaml.safe_load(f)
