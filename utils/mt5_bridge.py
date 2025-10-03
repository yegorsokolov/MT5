"""Utilities for loading the MetaTrader5 module across native and Wine bridges.

The bridge discovery helpers consult values in the following order:

1. Process environment variables.
2. Key/value pairs defined in the project ``.env`` file.
3. Hints extracted from ``LOGIN_INSTRUCTIONS_WINE.txt``.
"""

from __future__ import annotations

import importlib
import logging
import os
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOGIN_INSTRUCTIONS = PROJECT_ROOT / "LOGIN_INSTRUCTIONS_WINE.txt"
_DOTENV_PATH = PROJECT_ROOT / ".env"

_MT5_MODULE: Optional[ModuleType] = None
_BRIDGE_INFO: Dict[str, Any] = {}


class MetaTraderImportError(RuntimeError):
    """Raised when the MetaTrader5 module cannot be imported."""


def _read_login_instructions() -> list[str]:
    if not _LOGIN_INSTRUCTIONS.exists():
        return []
    try:
        return _LOGIN_INSTRUCTIONS.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:  # pragma: no cover - best effort only
        return []


def _normalize(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("\"") and cleaned.endswith("\""):
        return cleaned[1:-1]
    if cleaned.startswith("'") and cleaned.endswith("'"):
        return cleaned[1:-1]
    return cleaned


@lru_cache(maxsize=1)
def _load_dotenv() -> Dict[str, str]:
    if not _DOTENV_PATH.exists():
        return {}
    try:
        content = _DOTENV_PATH.read_text(encoding="utf-8", errors="ignore")
    except Exception:  # pragma: no cover - best effort only
        return {}

    data: Dict[str, str] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        data[key] = _normalize(value)
    return data


def _lookup_bridge_value(*keys: str) -> Optional[str]:
    for key in keys:
        value = os.getenv(key)
        if value:
            return _normalize(value)

    dotenv = _load_dotenv()
    for key in keys:
        value = dotenv.get(key)
        if value:
            return value

    return None


def _discover_windows_python_path() -> Optional[str]:
    keys = (
        "PYMT5LINUX_PYTHON",
        "PYMT5LINUX_WINDOWS_PYTHON",
        "WINE_PYTHON",
        "WIN_PYTHON",
        "WINE_PYTHON_PATH",
    )
    value = _lookup_bridge_value(*keys)
    if value:
        return value

    for line in _read_login_instructions():
        if line.lower().startswith("windows python"):
            _, _, remainder = line.partition(":")
            candidate = _normalize(remainder)
            if candidate and "<not detected>" not in candidate.lower():
                return candidate
    return None


def _discover_wine_prefix() -> Optional[str]:
    keys = (
        "PYMT5LINUX_WINEPREFIX",
        "WIN_PY_WINE_PREFIX",
        "WINE_PY_WINE_PREFIX",
        "MT5_WINE_PREFIX",
        "WINEPREFIX",
    )
    value = _lookup_bridge_value(*keys)
    if value:
        return value

    for line in _read_login_instructions():
        if line.lower().startswith("terminal prefix"):
            _, _, remainder = line.partition(":")
            candidate = _normalize(remainder)
            if candidate and "<not detected>" not in candidate.lower():
                return candidate
    return None


def _seed_bridge_environment() -> Dict[str, str]:
    """Populate environment variables commonly used by bridge loaders."""

    updates: Dict[str, str] = {}
    windows_python = _discover_windows_python_path()
    prefix = _discover_wine_prefix()

    if windows_python:
        for key in ("PYMT5LINUX_PYTHON", "PYMT5LINUX_WINDOWS_PYTHON", "WINE_PYTHON"):
            if not os.getenv(key):
                os.environ[key] = windows_python
                updates[key] = windows_python

    if prefix:
        for key in ("PYMT5LINUX_WINEPREFIX", "WIN_PY_WINE_PREFIX", "WINEPREFIX"):
            if not os.getenv(key):
                os.environ[key] = prefix
                updates[key] = prefix

    return updates


def _attempt_native_import() -> ModuleType:
    module = importlib.import_module("MetaTrader5")
    return module


def _attempt_bridge_import() -> Tuple[ModuleType, Dict[str, Any]]:
    errors: list[str] = []
    bridge = importlib.import_module("pymt5linux")
    info: Dict[str, Any] = {
        "backend": "pymt5linux",
        "bridge_module": getattr(bridge, "__file__", ""),
    }

    initializer = getattr(bridge, "initialize", None)
    if callable(initializer):
        try:
            initializer()
            info["initialized"] = True
        except Exception as exc:  # pragma: no cover - depends on Wine runtime
            info["initialize_error"] = str(exc)
            errors.append(f"initialize() failed: {exc}")

    candidate = getattr(bridge, "MetaTrader5", None)
    module: Optional[ModuleType] = None
    if isinstance(candidate, ModuleType):
        module = candidate
    elif callable(candidate):
        try:
            result = candidate()
        except Exception as exc:  # pragma: no cover - depends on bridge implementation
            errors.append(f"MetaTrader5() call failed: {exc}")
        else:
            if isinstance(result, ModuleType):
                module = result
            else:
                errors.append("MetaTrader5() did not return a module")

    if module is None:
        try:
            module = importlib.import_module("MetaTrader5")
        except Exception as exc:  # pragma: no cover - rely on aggregated message
            errors.append(str(exc))
            raise MetaTraderImportError(
                "MetaTrader5 import via pymt5linux failed: " + "; ".join(errors)
            ) from exc

    info["module_path"] = getattr(module, "__file__", "")
    if errors and "initialize_error" not in info:
        info["warnings"] = errors
    return module, info


def load_mt5_module(*, force: bool = False, prefer_bridge: bool = False) -> ModuleType:
    """Return the MetaTrader5 module using native or Wine bridge imports."""

    global _MT5_MODULE, _BRIDGE_INFO
    if _MT5_MODULE is not None and not force:
        return _MT5_MODULE

    attempts: list[str] = []
    _seed_bridge_environment()

    if not prefer_bridge:
        try:
            module = _attempt_native_import()
        except Exception as exc:
            attempts.append(f"native import failed: {exc}")
        else:
            _MT5_MODULE = module
            _BRIDGE_INFO = {
                "backend": "native",
                "module_path": getattr(module, "__file__", ""),
            }
            return module

    try:
        module, info = _attempt_bridge_import()
    except Exception as exc:
        attempts.append(str(exc))
        message = "; ".join(attempts)
        raise MetaTraderImportError(
            "Unable to import MetaTrader5 via any backend" + (f": {message}" if message else "")
        ) from exc

    _MT5_MODULE = module
    _BRIDGE_INFO = info
    logger.debug("MetaTrader5 module loaded via bridge: %s", info)
    return module


def describe_backend() -> Dict[str, Any]:
    """Return metadata describing how the MetaTrader5 module was loaded."""

    return dict(_BRIDGE_INFO)


def reload_mt5_module(prefer_bridge: bool = False) -> ModuleType:
    """Force a reload of the MetaTrader5 module."""

    global _MT5_MODULE
    _MT5_MODULE = None
    return load_mt5_module(force=True, prefer_bridge=prefer_bridge)
