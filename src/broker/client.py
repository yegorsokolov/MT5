"""Broker client that utilises account settings."""
from __future__ import annotations

from typing import Dict, Optional

from utils.config import get_account_settings


class BrokerClient:
    """Simple broker client initialising from configuration."""

    def __init__(self, settings: Optional[Dict[str, str]] = None) -> None:
        self.settings = settings or get_account_settings()
        self.connected = False

    def connect(self) -> None:
        """Pretend to connect to the broker using configured settings."""
        # In real implementation, use the API key/secret and endpoint to
        # establish a connection. Here we simply mark the client as connected.
        self.connected = True
        # Debug information could be logged here if needed.

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"BrokerClient(environment={self.settings.get('environment')!r})"
