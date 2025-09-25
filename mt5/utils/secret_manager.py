"""Fallback secret manager used in lightweight test environments."""

from __future__ import annotations

import os
from typing import Optional


class SecretManager:
    """Minimal secret manager that proxies lookups to environment variables."""

    def get_secret(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return os.getenv(name, default)
