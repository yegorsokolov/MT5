from __future__ import annotations

import base64
import os
from typing import Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_secret_mgr", Path(__file__).resolve().parent / "utils" / "secret_manager.py"
)
_secret_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_secret_mod)
SecretManager = _secret_mod.SecretManager


def _load_key(name: str, default: Optional[str] = None) -> bytes:
    """Return the AES key for ``name`` from ``SecretManager``.

    Keys are expected to be base64 encoded. ``default`` may provide a fallback
    value when the secret is not found.
    """
    sm = SecretManager()
    val = sm.get_secret(name, default=default)
    if not val:
        raise ValueError(f"Missing secret {name}")
    try:
        return base64.b64decode(val)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid base64 for {name}") from exc


def encrypt(data: bytes, key: bytes) -> bytes:
    """Encrypt ``data`` using ``key`` with AES-GCM."""
    aes = AESGCM(key)
    nonce = os.urandom(12)
    return nonce + aes.encrypt(nonce, data, None)


def decrypt(blob: bytes, key: bytes) -> bytes:
    """Decrypt ``blob`` using ``key`` with AES-GCM."""
    aes = AESGCM(key)
    nonce, ct = blob[:12], blob[12:]
    return aes.decrypt(nonce, ct, None)


__all__ = ["encrypt", "decrypt", "_load_key"]
