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

_DEFAULT_BACKEND = "auto"
_WINE_BACKENDS = {"wine", "pymt5linux", "mt5linux"}
_NATIVE_BACKENDS = {"native"}
_MT5LINUX_DEFAULT_HOST = "127.0.0.1"
_MT5LINUX_DEFAULT_PORT = 18812


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


def _lookup_bridge_value_with_source(*keys: str) -> tuple[Optional[str], Optional[str]]:
    for key in keys:
        if key in os.environ:
            value = os.environ.get(key)
            if value:
                return _normalize(value), f"env:{key}"

    dotenv = _load_dotenv()
    for key in keys:
        value = dotenv.get(key)
        if value:
            return value, f".env:{key}"

    return None, None


def _sanitize_env_key(value: str) -> str:
    sanitized: list[str] = []
    for char in value:
        if char.isalnum():
            sanitized.append(char.upper())
        else:
            sanitized.append("_")
    return "".join(sanitized)


def get_configured_backend(default: str = _DEFAULT_BACKEND) -> str:
    """Return the configured bridge backend preference."""

    value = _lookup_bridge_value("MT5_BRIDGE_BACKEND")
    if value is None:
        return default
    cleaned = value.strip()
    return cleaned or default


def _normalize_backend_choice(value: Optional[str]) -> Tuple[str, str]:
    if value is None:
        return _DEFAULT_BACKEND, _DEFAULT_BACKEND
    cleaned = value.strip() or _DEFAULT_BACKEND
    normalized = cleaned.lower()
    if normalized in {"", "default"}:
        return _DEFAULT_BACKEND, _DEFAULT_BACKEND
    if normalized in _WINE_BACKENDS:
        return cleaned, "wine"
    if normalized in _NATIVE_BACKENDS:
        return cleaned, "native"
    if normalized == _DEFAULT_BACKEND:
        return cleaned, _DEFAULT_BACKEND
    return cleaned, normalized


def _split_module_spec(spec: str) -> Tuple[str, Optional[str]]:
    module_name, _, attr = spec.partition(":")
    module_name = module_name.strip()
    attr = attr.strip() or None
    return module_name, attr


def _resolve_backend_module_spec(backend_raw: str, backend_key: str) -> Tuple[str, Optional[str]]:
    override_keys: list[str] = []
    if backend_raw:
        override_keys.append(f"MT5_BRIDGE_MODULE_{_sanitize_env_key(backend_raw)}")
    if backend_key and backend_key != backend_raw:
        override_keys.append(f"MT5_BRIDGE_MODULE_{_sanitize_env_key(backend_key)}")
    override_keys.append("MT5_BRIDGE_MODULE")

    spec = _lookup_bridge_value(*override_keys)
    if spec:
        return _split_module_spec(spec)

    alias_map = {
        "wine": "utils.bridge_clients.mt5linux_client",
        "mt5linux": "utils.bridge_clients.mt5linux_client",
        "pymt5linux": "utils.bridge_clients.mt5linux_client",
    }
    module_name = alias_map.get(backend_key.lower()) if backend_key else None
    if not module_name and backend_raw:
        module_name = alias_map.get(backend_raw.lower())
    if module_name:
        return module_name, None

    return _split_module_spec(backend_raw)


def get_mt5linux_connection_settings() -> Dict[str, Any]:
    """Return mt5linux connection settings derived from environment hints."""

    host_value, host_source = _lookup_bridge_value_with_source(
        "MT5LINUX_HOST",
        "PYMT5LINUX_HOST",
        "MT5_BRIDGE_HOST",
    )
    port_value, port_source = _lookup_bridge_value_with_source(
        "MT5LINUX_PORT",
        "PYMT5LINUX_PORT",
        "MT5_BRIDGE_PORT",
    )
    timeout_value, timeout_source = _lookup_bridge_value_with_source(
        "MT5LINUX_TIMEOUT",
        "MT5LINUX_TIMEOUT_SECONDS",
    )

    settings: Dict[str, Any] = {
        "host": host_value or _MT5LINUX_DEFAULT_HOST,
        "host_source": host_source or "default",
        "port_source": port_source or "default",
    }

    if port_value is None:
        settings["port"] = _MT5LINUX_DEFAULT_PORT
    else:
        settings["port_raw"] = port_value
        try:
            settings["port"] = int(port_value)
        except ValueError:
            settings["port_error"] = f"Invalid mt5linux port value: {port_value}"

    if timeout_value:
        settings["timeout_source"] = timeout_source or "default"
        settings["timeout_raw"] = timeout_value
        try:
            settings["timeout"] = float(timeout_value)
        except ValueError:
            settings["timeout_error"] = f"Invalid mt5linux timeout value: {timeout_value}"

    return settings


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


def _extract_module_from_candidate(
    candidate: Any,
    *,
    name: str,
    errors: list[str],
) -> Optional[ModuleType]:
    if candidate is None:
        return None
    if isinstance(candidate, ModuleType):
        return candidate
    if callable(candidate):
        try:
            result = candidate()
        except Exception as exc:  # pragma: no cover - depends on backend implementation
            errors.append(f"{name}() call failed: {exc}")
            return None
        if isinstance(result, ModuleType):
            return result
        errors.append(f"{name}() did not return a module")
    return None


def _attempt_bridge_import(
    module_name: str,
    *,
    backend: str,
    configured_backend: str,
    attr: Optional[str] = None,
) -> Tuple[ModuleType, Dict[str, Any]]:
    errors: list[str] = []
    try:
        bridge = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - relies on runtime availability
        raise MetaTraderImportError(
            f"Unable to import bridge backend '{configured_backend}' ({module_name}): {exc}"
        ) from exc

    info: Dict[str, Any] = {
        "backend": backend,
        "configured_backend": configured_backend,
        "bridge_package": module_name,
        "bridge_module": getattr(bridge, "__file__", ""),
    }

    initializer = getattr(bridge, "initialize", None)
    if callable(initializer):
        try:
            initializer()
            info["initialized"] = True
        except Exception as exc:  # pragma: no cover - depends on runtime
            info["initialize_error"] = str(exc)
            errors.append(f"initialize() failed: {exc}")

    bridge_info_getter = getattr(bridge, "bridge_info", None)
    if callable(bridge_info_getter):
        try:
            bridge_details = bridge_info_getter()
        except Exception as exc:  # pragma: no cover - depends on backend
            errors.append(f"bridge_info() failed: {exc}")
        else:
            if isinstance(bridge_details, dict):
                info.update({k: v for k, v in bridge_details.items() if v is not None})

    module: Optional[ModuleType] = None
    if attr:
        candidate = getattr(bridge, attr, None)
        module = _extract_module_from_candidate(candidate, name=attr, errors=errors)
        if module is None:
            errors.append(f"Attribute '{attr}' did not yield a MetaTrader5 module")
    else:
        for candidate_name in ("MetaTrader5", "load_mt5", "load", "get_mt5_module"):
            module = _extract_module_from_candidate(
                getattr(bridge, candidate_name, None),
                name=candidate_name,
                errors=errors,
            )
            if module is not None:
                info["bridge_entry"] = candidate_name
                break

    if module is None:
        try:
            module = importlib.import_module("MetaTrader5")
        except Exception as exc:  # pragma: no cover - rely on aggregated message
            errors.append(str(exc))
            raise MetaTraderImportError(
                "MetaTrader5 import via {0} failed: {1}".format(configured_backend, "; ".join(errors))
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
    configured = get_configured_backend()
    configured_raw, backend_key = _normalize_backend_choice(configured)

    attempt_native = backend_key == "native" or (backend_key == "auto" and not prefer_bridge)

    if attempt_native:
        try:
            module = _attempt_native_import()
        except Exception as exc:
            attempts.append(f"native import failed: {exc}")
            if backend_key == "native":
                message = "; ".join(attempts)
                raise MetaTraderImportError(
                    "MetaTrader5 import via native backend failed" + (f": {message}" if message else "")
                ) from exc
        else:
            _MT5_MODULE = module
            _BRIDGE_INFO = {
                "backend": "native",
                "configured_backend": configured_raw,
                "requested_backend": configured_raw,
                "module_path": getattr(module, "__file__", ""),
            }
            return module

    if backend_key == "auto":
        bridge_backend = "wine"
        module_name, attr = _resolve_backend_module_spec("wine", "wine")
        configured_descriptor = configured_raw if configured_raw != _DEFAULT_BACKEND else bridge_backend
    elif backend_key == "wine":
        bridge_backend = "wine"
        module_name, attr = _resolve_backend_module_spec(configured_raw, "wine")
        configured_descriptor = configured_raw
    else:
        bridge_backend = backend_key or configured_raw
        module_name, attr = _resolve_backend_module_spec(configured_raw, backend_key)
        configured_descriptor = configured_raw

    try:
        module, info = _attempt_bridge_import(
            module_name,
            backend=bridge_backend,
            configured_backend=configured_descriptor,
            attr=attr,
        )
    except Exception as exc:
        attempts.append(str(exc))
        message = "; ".join(attempts)
        raise MetaTraderImportError(
            "Unable to import MetaTrader5 via any backend" + (f": {message}" if message else "")
        ) from exc

    if bridge_backend == "wine":
        info.setdefault("mt5linux", get_mt5linux_connection_settings())

    info.setdefault("requested_backend", configured_raw)
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
