import argparse
import asyncio
import importlib
import json
import logging
import os
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from importlib import metadata
from importlib.util import find_spec

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - handled later
    psutil = None  # type: ignore

try:
    from packaging.markers import Marker
except Exception:  # pragma: no cover - packaging is optional
    Marker = None  # type: ignore

try:
    from . import load_config_data, save_config
except Exception:  # pragma: no cover - handled later
    load_config_data = None  # type: ignore
    save_config = None  # type: ignore

try:
    from mt5.config_models import AppConfig
except Exception:  # pragma: no cover - handled later
    AppConfig = None  # type: ignore

try:
    from utils import mt5_bridge
    from utils.mt5_bridge import (
        MetaTraderImportError,
        describe_backend as describe_mt5_backend,
        get_configured_backend,
        load_mt5_module,
    )
except Exception:  # pragma: no cover - bridge helper optional in minimal builds
    MetaTraderImportError = RuntimeError  # type: ignore

    def describe_mt5_backend() -> dict[str, Any]:  # type: ignore
        return {}

    def get_configured_backend(default: str = "auto") -> str:  # type: ignore
        return default

    def load_mt5_module():  # type: ignore
        import MetaTrader5 as _mt5  # type: ignore

        return _mt5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQ_FILE = PROJECT_ROOT / "requirements.txt"
CONFIG_FILE = Path(os.getenv("CONFIG_FILE", PROJECT_ROOT / "config.yaml"))
_WINE_BRIDGE_HINT_ENV_VARS = (
    "WINE_PYTHON",
    "WIN_PYTHON",
    "WINE_PYTHON_PATH",
    "PYMT5LINUX_PYTHON",
    "PYMT5LINUX_WINDOWS_PYTHON",
    "WIN_PY_WINE_PREFIX",
    "WINE_PY_WINE_PREFIX",
    "MT5_WINE_PREFIX",
    "PYMT5LINUX_WINEPREFIX",
    "WINEPREFIX",
    "MT5LINUX_HOST",
    "MT5LINUX_PORT",
    "MT5LINUX_TIMEOUT",
    "MT5LINUX_TIMEOUT_SECONDS",
)
AUTO_INSTALL_DEPENDENCIES_DEFAULT = os.getenv("AUTO_INSTALL_DEPENDENCIES", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}

MIN_RAM_GB = 2
REC_RAM_GB = 8
MIN_CORES = 1
REC_CORES = 4

MIN_PYTHON = (3, 10)
MAX_PYTHON = (3, 12)
RECOMMENDED_PYTHON = "3.10"

_SPECIFIER_SPLIT_RE = re.compile(r"\s*(?:==|!=|<=|>=|~=|===|<|>|=)")


def _python_major_minor() -> tuple[int, int]:
    """Return the active interpreter's ``(major, minor)`` version tuple."""

    version = sys.version_info
    return version.major, version.minor


def _marker_allows(marker_text: str | None) -> bool:
    if not marker_text:
        return True
    if Marker is not None:
        try:
            return Marker(marker_text).evaluate()
        except Exception:
            return True
    return True


class EnvironmentCheckError(RuntimeError):
    """Exception raised when environment diagnostics fail."""

    def __init__(
        self,
        message: str,
        *,
        missing_dependencies: Iterable[str] | None = None,
        install_attempts: dict[str, str] | None = None,
        hardware: dict[str, Any] | None = None,
        manual_tests: Iterable[str] | None = None,
        checks: Iterable[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.missing_dependencies = list(missing_dependencies or [])
        self.install_attempts = dict(install_attempts or {})
        self.hardware = dict(hardware or {})
        self.manual_tests = list(manual_tests or [])
        self.check_results = list(checks or [])


def _parse_requirement_line(line: str) -> tuple[str, str] | None:
    """Extract the requirement and canonical module name from a line."""

    if not line:
        return None

    candidate = line.split("#", 1)[0].strip()
    if not candidate:
        return None

    if candidate.startswith("#"):
        return None

    marker_text: str | None = None
    if ";" in candidate:
        candidate, marker_text = candidate.split(";", 1)
        candidate = candidate.strip()
        marker_text = marker_text.strip() or None
        if not _marker_allows(marker_text):
            return None

    if not candidate or candidate.startswith("-"):
        return None

    candidate = candidate.split("@", 1)[0].strip()
    if not candidate:
        return None

    candidate = candidate.replace("(", " ").replace(")", " ")

    if "[" in candidate:
        candidate = candidate.split("[", 1)[0].strip()
        if not candidate:
            return None

    base_name = _SPECIFIER_SPLIT_RE.split(candidate, 1)[0].strip().strip("()")
    if not base_name or base_name.startswith("#"):
        return None

    module_name = base_name.replace("-", "_")
    if not module_name:
        return None

    return base_name, module_name


def _distribution_installed(distribution_name: str, module_name: str) -> bool:
    """Return ``True`` when a package appears to be installed."""

    try:
        metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError:
        pass
    except Exception:  # pragma: no cover - metadata lookup should not fail
        # Fall back to module level introspection in unexpected scenarios.
        return find_spec(module_name) is not None
    else:
        return True

    return find_spec(module_name) is not None


def _check_dependencies() -> list[str]:
    missing: list[str] = []
    if not REQ_FILE.exists():
        return missing

    for line in REQ_FILE.read_text().splitlines():
        parsed = _parse_requirement_line(line)
        if parsed is None:
            continue
        pkg_name, module_name = parsed
        if not _distribution_installed(pkg_name, module_name):
            missing.append(pkg_name)
    return missing


def _collect_missing_dependencies() -> list[str]:
    missing = _check_dependencies()
    if "MetaTrader5" in missing:
        if sys.platform.startswith("win"):
            pass
        elif any(os.getenv(key) for key in _WINE_BRIDGE_HINT_ENV_VARS):
            missing = [pkg for pkg in missing if pkg != "MetaTrader5"]
        else:
            instructions = PROJECT_ROOT / "LOGIN_INSTRUCTIONS_WINE.txt"
            if instructions.exists():
                missing = [pkg for pkg in missing if pkg != "MetaTrader5"]
    if find_spec("psutil") is None:
        missing.append("psutil")
    return sorted(set(missing))


def _mt5_backend_suffix() -> str:
    info = describe_mt5_backend()
    backend = info.get("backend") if isinstance(info, dict) else None
    if backend:
        return f" (backend: {backend})"
    return ""


def _attempt_dependency_install(packages: Iterable[str]) -> dict[str, str]:
    """Attempt to install the provided packages using ``pip``.

    Returns a mapping of package name to installation status.
    """

    results: dict[str, str] = OrderedDict()
    for package in packages:
        cmd = [sys.executable, "-m", "pip", "install", package]
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            results[package] = f"error: {exc}"
            continue

        if completed.returncode == 0:
            results[package] = "installed"
        else:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            diagnostic = stderr or stdout or "installation failed"
            results[package] = f"failed: {diagnostic.splitlines()[0]}"

    return results


def _mt5_terminal_downloaded() -> bool:
    """Best-effort check that an MT5 terminal binary is available."""

    env_path = os.getenv("MT5_TERMINAL_PATH")
    candidate_paths = [
        Path(env_path) if env_path else None,
        PROJECT_ROOT / "mt5",
        PROJECT_ROOT / "MT5",
    ]

    for path in candidate_paths:
        if path and path.exists():
            return True
    return False


def _detect_git_credentials() -> bool:
    """Check that Git can access a configured remote (best effort)."""

    git_dir = PROJECT_ROOT / ".git"
    if not git_dir.exists():
        return False

    config_file = git_dir / "config"
    if not config_file.exists():
        return False

    return "url" in config_file.read_text().lower()


def _has_env_file() -> bool:
    """Check for the presence of a .env or config file."""

    candidates = [
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / "config.env",
        PROJECT_ROOT / "config/.env",
    ]

    return any(path.exists() for path in candidates)


def _normalise_env_value(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1]
    if "\\" in cleaned:
        try:
            cleaned = bytes(cleaned, "utf-8").decode("unicode_escape")
        except Exception:
            return cleaned
    return cleaned


def _read_env_pairs(path: Path) -> dict[str, str]:
    """Return key/value pairs from ``path`` if the file exists."""

    try:
        from deployment.runtime_secrets import _read_env_file as _secrets_read_env
    except Exception:
        raw_values: dict[str, str] = {}
        if not path.exists():
            return raw_values
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                raw_values[key.strip()] = value.strip()
    else:
        raw_values = _secrets_read_env(path)

    normalised: dict[str, str] = {}
    for key, value in raw_values.items():
        normalised[key] = _normalise_env_value(str(value))
    return normalised


def _check_mt5_login() -> dict[str, Any]:
    """Verify that the MetaTrader 5 terminal reports an authenticated session."""

    instruction = (
        "Launch the MT5 terminal and verify the bot can log in using your broker credentials."
    )
    if not _mt5_terminal_downloaded():
        return {
            "name": "MetaTrader 5 login",
            "status": "failed",
            "detail": "MetaTrader 5 terminal not found. Set MT5_TERMINAL_PATH or place the terminal under 'mt5/'.",
            "followup": instruction,
        }

    try:
        mt5 = load_mt5_module()
        from brokers import mt5_direct
    except (MetaTraderImportError, ImportError, RuntimeError) as exc:  # pragma: no cover
        return {
            "name": "MetaTrader 5 login",
            "status": "failed",
            "detail": f"MetaTrader5 Python bindings unavailable: {exc}",
            "followup": instruction,
        }

    try:
        logged_in = mt5_direct.is_terminal_logged_in()
    except Exception as exc:  # pragma: no cover - terminal errors vary
        return {
            "name": "MetaTrader 5 login",
            "status": "failed",
            "detail": f"MetaTrader5 login check failed: {exc}",
            "followup": instruction,
        }

    if logged_in:
        return {
            "name": "MetaTrader 5 login",
            "status": "passed",
            "detail": f"MetaTrader 5 terminal login detected{_mt5_backend_suffix()}.",
            "followup": None,
        }
    return {
        "name": "MetaTrader 5 login",
        "status": "failed",
        "detail": "MetaTrader 5 terminal not logged in.",
        "followup": instruction,
    }


def _check_mt5_ping() -> dict[str, Any]:
    """Ping the MetaTrader 5 terminal via the Python bridge."""

    instruction = (
        "From the running bot, execute a simple ping to the MT5 terminal to confirm connectivity."
    )
    try:
        mt5 = load_mt5_module()
        from brokers import mt5_direct
    except (MetaTraderImportError, ImportError, RuntimeError) as exc:  # pragma: no cover
        return {
            "name": "MetaTrader 5 connectivity",
            "status": "failed",
            "detail": f"MetaTrader5 Python bindings unavailable: {exc}",
            "followup": instruction,
        }

    try:
        if not mt5_direct.initialize():
            return {
                "name": "MetaTrader 5 connectivity",
                "status": "failed",
                "detail": "Failed to initialise MetaTrader 5 connection.",
                "followup": instruction,
            }
        info = mt5.terminal_info()
        version_fn = getattr(mt5, "version", None)
        version = ""
        if callable(version_fn):
            try:
                version_tuple = version_fn()
                if isinstance(version_tuple, (tuple, list)) and version_tuple:
                    version = f" build {version_tuple[-1]}"
            except Exception:
                version = ""
    except Exception as exc:  # pragma: no cover - terminal errors vary
        return {
            "name": "MetaTrader 5 connectivity",
            "status": "failed",
            "detail": f"MetaTrader5 connectivity check failed: {exc}",
            "followup": instruction,
        }
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass

    if info is None:
        return {
            "name": "MetaTrader 5 connectivity",
            "status": "failed",
            "detail": "MetaTrader5.terminal_info returned no data.",
            "followup": instruction,
        }

    connected = bool(getattr(info, "connected", True))
    if connected:
        return {
            "name": "MetaTrader 5 connectivity",
            "status": "passed",
            "detail": f"Terminal responded successfully{version}{_mt5_backend_suffix()}.",
            "followup": None,
        }
    return {
        "name": "MetaTrader 5 connectivity",
        "status": "failed",
        "detail": "Terminal reported a disconnected state.",
        "followup": instruction,
    }


def _normalize_windows_path(path: str) -> str:
    """Return ``path`` without surrounding whitespace or quotes."""

    cleaned = path.strip()
    if cleaned.startswith("\"") and cleaned.endswith("\""):
        return cleaned[1:-1]
    if cleaned.startswith("'") and cleaned.endswith("'"):
        return cleaned[1:-1]
    return cleaned


def _read_login_instructions() -> list[str]:
    instructions = PROJECT_ROOT / "LOGIN_INSTRUCTIONS_WINE.txt"
    if not instructions.exists():
        return []
    try:
        return instructions.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:  # pragma: no cover - defensive best effort
        return []


def _discover_wine_python_path() -> tuple[str | None, str | None]:
    """Return the configured Wine-hosted Python interpreter if available."""

    for key in ("WINE_PYTHON", "WIN_PYTHON", "WINE_PYTHON_PATH"):
        value = os.getenv(key)
        if value:
            return _normalize_windows_path(value), f"environment variable {key}"

    for line in _read_login_instructions():
        if line.lower().startswith("windows python"):
            _, _, remainder = line.partition(":")
            candidate = _normalize_windows_path(remainder)
            if candidate and "<not detected>" not in candidate.lower():
                return candidate, "LOGIN_INSTRUCTIONS_WINE.txt"

    return None, None


def _discover_wine_prefix() -> tuple[str | None, str | None]:
    """Return the Wine prefix used for the MetaTrader bridge."""

    for key in ("WIN_PY_WINE_PREFIX", "WINE_PY_WINE_PREFIX", "MT5_WINE_PREFIX", "WINEPREFIX"):
        value = os.getenv(key)
        if value:
            return value, f"environment variable {key}"

    for line in _read_login_instructions():
        if line.lower().startswith("terminal prefix"):
            _, _, remainder = line.partition(":")
            candidate = remainder.strip()
            if candidate and "<not detected>" not in candidate.lower():
                return candidate, "LOGIN_INSTRUCTIONS_WINE.txt"

    return None, None


def _discover_mt5linux_aux_python() -> tuple[Path | None, str | None]:
    """Return the Python interpreter from the auxiliary mt5linux virtualenv."""

    candidates: list[tuple[Path, str]] = []
    env_path = os.getenv("MT5LINUX_VENV_PATH")
    if env_path:
        candidates.append((Path(env_path), "MT5LINUX_VENV_PATH"))

    default_path = PROJECT_ROOT / ".mt5linux-venv"
    candidates.append((default_path, "project default"))

    for base, source in candidates:
        if not base:
            continue
        scripts_dir = "Scripts" if os.name == "nt" else "bin"
        python_name = "python.exe" if os.name == "nt" else "python"
        candidate = base / scripts_dir / python_name
        if candidate.exists():
            return candidate, source

    return None, None


def _check_wine_bridge() -> dict[str, Any]:
    """Validate the mt5linux bridge by connecting to the RPyC server."""

    name = "mt5linux bridge"
    instruction = (
        "Ensure the mt5linux RPyC server is running (rerun install_programmatic_bridge.sh or restart the bridge service)."
    )

    backend_pref = get_configured_backend()
    backend_alias = backend_pref.strip().lower().split(":", 1)[0]

    if sys.platform.startswith("win"):
        return {
            "name": name,
            "status": "skipped",
            "detail": "Native Windows runtime detected; mt5linux bridge validation not required.",
            "followup": None,
        }

    if backend_alias in {"native"}:
        detail = "mt5linux bridge validation skipped because MT5_BRIDGE_BACKEND is set to a native loader"
        if backend_pref:
            detail += f" ({backend_pref})."
        return {
            "name": name,
            "status": "skipped",
            "detail": detail,
            "followup": None,
        }

    if backend_alias not in {"auto", "wine", "pymt5linux", "mt5linux"}:
        detail = "Configured MetaTrader backend does not rely on the mt5linux bridge."
        if backend_pref:
            detail += f" (MT5_BRIDGE_BACKEND={backend_pref})."
        return {
            "name": name,
            "status": "skipped",
            "detail": detail,
            "followup": None,
        }

    settings = mt5_bridge.get_mt5linux_connection_settings()
    if settings.get("port_error"):
        return {
            "name": name,
            "status": "failed",
            "detail": settings["port_error"],
            "followup": instruction,
        }
    if settings.get("timeout_error"):
        return {
            "name": name,
            "status": "failed",
            "detail": settings["timeout_error"],
            "followup": instruction,
        }

    host = settings["host"]
    port = settings.get("port", 0)
    timeout = settings.get("timeout", 30.0)

    mt5linux_import_error: Exception | None = None
    try:
        importlib.import_module("mt5linux")
    except Exception as exc:  # pragma: no cover - optional dependency
        mt5linux_import_error = exc

    if mt5linux_import_error is not None:
        aux_python, aux_source = _discover_mt5linux_aux_python()
        if aux_python is not None:
            try:
                subprocess.run(
                    [str(aux_python), "-c", "import mt5linux"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except Exception as aux_exc:  # pragma: no cover - auxiliary validation is best-effort
                detail = (
                    "mt5linux package unavailable in active interpreter. "
                    f"Auxiliary environment check ({aux_source}) failed: {aux_exc}. "
                    "Activate the helper via ./use-mt5linux.sh before rerunning."
                )
                return {
                    "name": name,
                    "status": "failed",
                    "detail": detail,
                    "followup": instruction,
                }
        else:
            detail = (
                "mt5linux package unavailable in active interpreter and auxiliary virtualenv not detected. "
                "Run ./use-mt5linux.sh to create the bridge client environment."
            )
            return {
                "name": name,
                "status": "failed",
                "detail": detail,
                "followup": instruction,
            }

    try:
        import rpyc  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        return {
            "name": name,
            "status": "failed",
            "detail": f"rpyc dependency missing: {exc}",
            "followup": instruction,
        }

    config = {
        "sync_request_timeout": timeout,
        "connection_timeout": timeout,
        "allow_public_attrs": True,
    }

    try:
        conn = rpyc.classic.connect(host, port=port, config=config)
    except OSError as exc:
        return {
            "name": name,
            "status": "failed",
            "detail": f"Unable to connect to mt5linux at {host}:{port}: {exc}",
            "followup": instruction,
        }

    try:
        try:
            version = conn.eval("import MetaTrader5 as mt5; getattr(mt5, '__version__', '')")
        except Exception as exc:
            return {
                "name": name,
                "status": "failed",
                "detail": f"MetaTrader5 import via mt5linux failed: {exc}",
                "followup": instruction,
            }
    finally:
        try:
            conn.close()
        except Exception:
            pass

    version_text = f"MetaTrader5 {version}" if version else "MetaTrader5 module available"
    host_source = settings.get("host_source")
    port_source = settings.get("port_source")

    detail_parts = [f"Connected to mt5linux at {host}:{port}", version_text]
    if host_source and host_source != "default":
        detail_parts.append(f"host from {host_source}")
    if port_source and port_source != "default":
        detail_parts.append(f"port from {port_source}")
    if timeout and settings.get("timeout_source") not in {None, "default"}:
        detail_parts.append(f"timeout from {settings['timeout_source']}")

    detail = "; ".join(detail_parts)
    return {
        "name": name,
        "status": "passed",
        "detail": detail,
        "followup": None,
    }


def _check_git_remote() -> dict[str, Any]:
    """Validate that Git credentials can access the configured remote."""

    instruction = (
        "Validate that Git operations (clone/pull/push) succeed using the configured credentials."
    )
    if not _detect_git_credentials():
        return {
            "name": "Git remote access",
            "status": "failed",
            "detail": "No Git remote configured; run 'git remote add origin ...' and ensure credentials are valid.",
            "followup": instruction,
        }

    try:
        completed = subprocess.run(
            ["git", "ls-remote", "--exit-code", "origin", "HEAD"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return {
            "name": "Git remote access",
            "status": "failed",
            "detail": "git executable not found in PATH.",
            "followup": instruction,
        }
    except subprocess.TimeoutExpired:
        return {
            "name": "Git remote access",
            "status": "failed",
            "detail": "git ls-remote timed out; check network connectivity.",
            "followup": instruction,
        }

    if completed.returncode == 0:
        return {
            "name": "Git remote access",
            "status": "passed",
            "detail": "Verified access to the origin remote.",
            "followup": None,
        }

    diagnostic = completed.stderr.strip() or completed.stdout.strip() or "git ls-remote failed"
    return {
        "name": "Git remote access",
        "status": "failed",
        "detail": diagnostic.splitlines()[0],
        "followup": instruction,
    }


def _check_env_loaded() -> dict[str, Any]:
    """Confirm that key/value pairs from .env are visible to the process."""

    instruction = "Check that environment variables from the .env file are loaded before starting the bot."
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return {
            "name": ".env availability",
            "status": "failed",
            "detail": "No .env file found at the project root.",
            "followup": instruction,
        }

    values = _read_env_pairs(env_path)
    if not values:
        return {
            "name": ".env availability",
            "status": "failed",
            "detail": ".env file contains no key/value pairs.",
            "followup": instruction,
        }

    missing = [key for key, value in values.items() if os.environ.get(key) != value]
    if missing:
        preview = ", ".join(missing[:5])
        more = "" if len(missing) <= 5 else f" (and {len(missing) - 5} more)"
        return {
            "name": ".env availability",
            "status": "failed",
            "detail": f"Environment missing exports for: {preview}{more}.",
            "followup": instruction,
        }

    return {
        "name": ".env availability",
        "status": "passed",
        "detail": f"Loaded {len(values)} environment variables from .env.",
        "followup": None,
    }


def _check_oracle_pipeline() -> dict[str, Any]:
    """Exercise the oracle scalper pipeline."""

    instruction = (
        "Exercise the oracle scalper pipeline (e.g. collect() then assess_probabilities()) to confirm external market data APIs respond."
    )
    try:
        from oracles.oracle_scalper import OracleScalper
    except Exception as exc:  # pragma: no cover - optional dependency
        return {
            "name": "Oracle scalper pipeline",
            "status": "failed",
            "detail": f"oracle_scalper unavailable: {exc}",
            "followup": instruction,
        }

    scalper = OracleScalper()
    try:
        symbols = ["EURUSD", "BTC", "ETH", "XAUUSD"]
        events = scalper.collect(symbols)
        summary = scalper.assess_probabilities(events)
    except Exception as exc:  # pragma: no cover - network errors vary
        return {
            "name": "Oracle scalper pipeline",
            "status": "failed",
            "detail": f"Oracle pipeline raised an exception: {exc}",
            "followup": instruction,
        }

    if not summary.empty:
        return {
            "name": "Oracle scalper pipeline",
            "status": "passed",
            "detail": f"Collected {len(summary)} probability rows from external oracles.",
            "followup": None,
        }

    return {
        "name": "Oracle scalper pipeline",
        "status": "failed",
        "detail": "Oracle pipeline returned no data; check API credentials and network access.",
        "followup": instruction,
    }


def _check_inference_service() -> dict[str, Any]:
    """Start the inference FastAPI service and query the /health endpoint."""

    instruction = (
        "Start the inference FastAPI service (services.inference_server) and hit the /health endpoint to verify REST API integrations."
    )
    try:
        from fastapi.testclient import TestClient  # type: ignore
        from services import inference_server
    except Exception as exc:  # pragma: no cover - optional dependency
        return {
            "name": "Inference service health",
            "status": "failed",
            "detail": f"FastAPI test client unavailable: {exc}",
            "followup": instruction,
        }

    try:
        with TestClient(inference_server.app) as client:
            response = client.get("/health")
    except Exception as exc:  # pragma: no cover - network/server errors vary
        return {
            "name": "Inference service health",
            "status": "failed",
            "detail": f"Failed to query /health: {exc}",
            "followup": instruction,
        }

    if response.status_code == 200 and response.json().get("status") == "ok":
        loaded = response.json().get("loaded_models", [])
        detail = "Inference service healthy"
        if loaded:
            detail += f" with {len(loaded)} loaded model(s)."
        return {
            "name": "Inference service health",
            "status": "passed",
            "detail": detail,
            "followup": None,
        }

    return {
        "name": "Inference service health",
        "status": "failed",
        "detail": f"Unexpected response: HTTP {response.status_code} {response.text}",
        "followup": instruction,
    }


def _check_feature_worker() -> dict[str, Any]:
    """Ensure the feature worker FastAPI app and resource monitor queues start."""

    instruction = (
        "Spin up the feature worker FastAPI app and ensure background tasks can subscribe to the message bus/broker queue."
    )
    try:
        from fastapi.testclient import TestClient  # type: ignore
        from services import feature_worker
        from utils.resource_monitor import monitor as resource_monitor
    except Exception as exc:  # pragma: no cover - optional dependency
        return {
            "name": "Feature worker service",
            "status": "failed",
            "detail": f"Feature worker dependencies unavailable: {exc}",
            "followup": instruction,
        }

    received: Any | None = None
    try:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            queue = resource_monitor.subscribe()
            loop.run_until_complete(queue.put("diagnostic"))
            received = loop.run_until_complete(queue.get())
        finally:
            asyncio.set_event_loop(None)
            loop.close()
            resource_monitor.stop()
    except Exception as exc:  # pragma: no cover - event loop failures vary
        return {
            "name": "Feature worker service",
            "status": "failed",
            "detail": f"Resource monitor subscription failed: {exc}",
            "followup": instruction,
        }

    if received != "diagnostic":
        return {
            "name": "Feature worker service",
            "status": "failed",
            "detail": "Message bus echo test failed.",
            "followup": instruction,
        }

    try:
        with TestClient(feature_worker.app) as client:
            response = client.post(
                "/compute",
                json={"symbol": "TEST", "start": "0", "end": "3"},
            )
    except Exception as exc:  # pragma: no cover - server errors vary
        return {
            "name": "Feature worker service",
            "status": "failed",
            "detail": f"Feature worker request failed: {exc}",
            "followup": instruction,
        }

    if response.status_code == 200 and isinstance(response.json(), list):
        return {
            "name": "Feature worker service",
            "status": "passed",
            "detail": "Feature worker responded to /compute and queue subscription succeeded.",
            "followup": None,
        }

    return {
        "name": "Feature worker service",
        "status": "failed",
        "detail": f"Unexpected /compute response: HTTP {response.status_code} {response.text}",
        "followup": instruction,
    }


def _check_python_runtime() -> dict[str, Any]:
    """Verify that the active interpreter matches supported versions."""

    version = sys.version_info
    runtime = f"{version.major}.{version.minor}.{version.micro}"
    instruction = (
        "Install a supported Python interpreter and rerun scripts/setup_ubuntu.sh. "
        f"Set MT5_PYTHON_SERIES={RECOMMENDED_PYTHON} (and optionally MT5_PYTHON_PATCH) before rerunning the automation to "
        "align with the default toolchain."
    )

    if version < MIN_PYTHON or version >= MAX_PYTHON:
        supported_min = f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
        if MIN_PYTHON[0] == MAX_PYTHON[0]:
            supported_max = f"{MAX_PYTHON[0]}.{MAX_PYTHON[1] - 1}"
        else:
            supported_max = f"{MAX_PYTHON[0] - 1}.{MAX_PYTHON[1]}"
        supported_range = f"{supported_min}.x through {supported_max}.x"
        return {
            "name": "python-runtime",
            "status": "failed",
            "detail": (
                "Python "
                f"{runtime} is not supported. The project requires Python "
                f"{supported_range}, with {RECOMMENDED_PYTHON}.x recommended for binary dependencies."
            ),
            "followup": instruction,
        }

    detail = (
        f"Python runtime detected: {runtime}. Recommended interpreter: {RECOMMENDED_PYTHON}.x."
    )

    followup = None

    if version.minor != int(RECOMMENDED_PYTHON.split(".")[1]):
        detail += (
            " Some third-party wheels may be missing; pin to the recommended version if package installs fail."
        )
        followup = (
            f"Set MT5_PYTHON_SERIES={RECOMMENDED_PYTHON} before running scripts/setup_ubuntu.sh to maximise wheel availability."
        )

    return {
        "name": "python-runtime",
        "status": "passed",
        "detail": detail,
        "followup": followup,
    }


def _check_distributed_dependencies() -> dict[str, Any]:
    """Confirm Ray, torch-geometric and uvloop are available when supported."""

    name = "Accelerated training dependencies"
    major, minor = _python_major_minor()
    ray_supported = major < 3 or (major == 3 and minor <= 12)
    uvloop_supported = not sys.platform.startswith("win")

    followup_pkgs: list[str] = ["'ray[default]'", "torch-geometric"]
    if uvloop_supported:
        followup_pkgs.append("uvloop")

    followup = (
        "Install Ray, torch-geometric and uvloop packages: pip install "
        + " ".join(followup_pkgs)
    )

    missing: list[str] = []
    detail_parts: list[str] = []

    if ray_supported:
        for pkg, module in (("ray", "ray"), ("torch-geometric", "torch_geometric")):
            if not _distribution_installed(pkg, module):
                missing.append(pkg)
    else:
        detail_parts.append(
            "Python "
            f"{major}.{minor} currently lacks official Ray/torch-geometric wheels. "
            "The distributed trainers will fall back to pure-Python implementations."
        )

    if uvloop_supported:
        if not _distribution_installed("uvloop", "uvloop"):
            missing.append("uvloop")
    else:
        detail_parts.append("uvloop is not supported on Windows and will be skipped.")

    if missing:
        display = ", ".join(missing)
        detail = f"Missing packages: {display}."
        if detail_parts:
            detail += " " + " ".join(detail_parts)
        return {
            "name": name,
            "status": "failed",
            "detail": detail,
            "followup": followup,
        }

    if ray_supported:
        detail = (
            "Ray, torch-geometric and uvloop available. Distributed and graph "
            "trainers are fully enabled."
        )
        return {
            "name": name,
            "status": "passed",
            "detail": detail,
            "followup": None,
        }

    detail = " ".join(detail_parts) or "Ray/torch-geometric skipped due to runtime."
    if uvloop_supported:
        detail = "uvloop available. " + detail
    return {
        "name": name,
        "status": "skipped",
        "detail": detail.strip(),
        "followup": None,
    }


def _check_model_configuration() -> dict[str, Any]:
    """Validate that model configuration references consistent artifacts."""

    instruction = (
        "Ensure config.yaml uses matching model_type entries and model paths for "
        "the tabular or AutoML trainers."
    )
    name = "Model configuration compatibility"

    if load_config_data is None or not CONFIG_FILE.exists():
        return {
            "name": name,
            "status": "passed",
            "detail": "Configuration loader unavailable or config.yaml missing; skipping model compatibility check.",
            "followup": None,
        }

    try:
        cfg = load_config_data(path=CONFIG_FILE, resolve_secrets=True)
    except Exception as exc:  # pragma: no cover - configuration errors vary
        return {
            "name": name,
            "status": "failed",
            "detail": f"Unable to load config.yaml: {exc}",
            "followup": "Resolve the configuration error before running the bot.",
        }

    def _extract_model_type(mapping: Mapping[str, Any] | None) -> str:
        if not isinstance(mapping, Mapping):
            return ""
        value = mapping.get("model_type")
        if value is None:
            model_section = mapping.get("model")
            if isinstance(model_section, Mapping):
                value = model_section.get("type")
        return str(value or "").lower()

    primary_type = _extract_model_type(cfg if isinstance(cfg, Mapping) else None)
    training_section = cfg.get("training") if isinstance(cfg, Mapping) else None
    training_type = _extract_model_type(training_section if isinstance(training_section, Mapping) else None)
    model_type_raw = training_type or primary_type
    model_type = "automl" if model_type_raw == "autogluon" else model_type_raw

    ensemble = None
    if isinstance(training_section, Mapping):
        ensemble = training_section.get("ensemble_models")
    if not ensemble and isinstance(cfg, Mapping):
        ensemble = cfg.get("ensemble_models")

    offending_paths: list[str] = []
    automl_aliases = {"automl", "autogluon"}
    if isinstance(ensemble, (list, tuple)):
        offending_paths = [
            str(item)
            for item in ensemble
            if isinstance(item, str)
            and any(alias in item.lower() for alias in automl_aliases)
            and model_type not in automl_aliases
        ]
    elif (
        isinstance(ensemble, str)
        and any(alias in ensemble.lower() for alias in automl_aliases)
        and model_type not in automl_aliases
    ):
        offending_paths = [ensemble]

    if offending_paths:
        preview = ", ".join(offending_paths[:3])
        if len(offending_paths) > 3:
            preview += f" (and {len(offending_paths) - 3} more)"
        return {
            "name": name,
            "status": "failed",
            "detail": (
                "Configuration references AutoML model paths but "
                "model_type is not set to 'automl': "
                f"{preview}."
            ),
            "followup": instruction,
        }

    if model_type == "automl":
        detail = "Configuration targets the FLAML AutoML ensemble."
    elif model_type == "tabular":
        detail = "Configuration targets the sklearn tabular pipeline."
    else:
        detail = "Configuration does not reference AutoML-specific paths."

    return {
        "name": name,
        "status": "passed",
        "detail": detail,
        "followup": None,
    }


_AUTOMATED_CHECKS: tuple[Callable[[], dict[str, Any]], ...] = (
    _check_python_runtime,
    _check_distributed_dependencies,
    _check_model_configuration,
    _check_mt5_login,
    _check_mt5_ping,
    _check_wine_bridge,
    _check_git_remote,
    _check_env_loaded,
    _check_oracle_pipeline,
    _check_inference_service,
    _check_feature_worker,
)


def _run_automated_preflight() -> tuple[list[dict[str, Any]], list[str]]:
    """Execute automated pre-flight checks and collect follow-up actions."""

    results: list[dict[str, Any]] = []
    manual: list[str] = []

    for check in _AUTOMATED_CHECKS:
        try:
            result = check()
        except Exception as exc:  # pragma: no cover - defensive guard
            result = {
                "name": getattr(check, "__name__", "check"),
                "status": "failed",
                "detail": f"Unexpected error while running automated check: {exc}",
                "followup": "Inspect logs to diagnose the failing automated check.",
            }
        results.append(result)
        status = (result.get("status") or "").lower()
        followup = result.get("followup")
        if status != "passed" and followup:
            manual.append(str(followup))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_manual = []
    for item in manual:
        if item not in seen:
            unique_manual.append(item)
            seen.add(item)

    return results, unique_manual


def _greater_than(value: Any, threshold: int) -> bool:
    try:
        return value is not None and value > threshold
    except TypeError:
        return False


def _adjust_config_for_low_spec() -> None:
    if not CONFIG_FILE.exists() or load_config_data is None or save_config is None:
        return

    raw_cfg: dict[str, Any] = load_config_data(
        path=CONFIG_FILE, resolve_secrets=False
    )
    resolved_cfg: dict[str, Any] = load_config_data(
        path=CONFIG_FILE, resolve_secrets=True
    )

    raw_training = raw_cfg.get("training")
    if not isinstance(raw_training, dict):
        raw_training = {}
        raw_cfg["training"] = raw_training

    resolved_training = resolved_cfg.get("training")
    if not isinstance(resolved_training, dict):
        resolved_training = {}
        resolved_cfg["training"] = resolved_training

    changed = False

    def _effective_training_value(key: str) -> Any:
        value = resolved_training.get(key)
        if value is None and key in resolved_cfg:
            return resolved_cfg.get(key)
        return value

    def _set_consistently(key: str, value: Any) -> None:
        nonlocal changed

        for mapping in (raw_training, resolved_training):
            if mapping.get(key) != value:
                mapping[key] = value
                changed = True

        if key in raw_cfg and raw_cfg.get(key) != value:
            raw_cfg[key] = value
            changed = True

        if key in resolved_cfg and resolved_cfg.get(key) != value:
            resolved_cfg[key] = value
            changed = True

    batch_size = _effective_training_value("batch_size")
    if batch_size is None or _greater_than(batch_size, 32):
        _set_consistently("batch_size", 32)

    n_jobs = _effective_training_value("n_jobs")
    if _greater_than(n_jobs, 1):
        _set_consistently("n_jobs", 1)

    if not changed:
        return

    if AppConfig is not None:
        AppConfig(**resolved_cfg)

    save_config(raw_cfg)


def ensure_environment(
    strict: bool | None = None,
    *,
    auto_install: bool | None = None,
) -> dict[str, Any]:
    """Validate dependencies, hardware and pre-run requirements.

    Parameters
    ----------
    strict:
        When ``True`` the function raises on hardware shortfalls. When
        ``False`` the function logs warnings for hardware checks. This does not
        affect dependency validation – required packages are always enforced.
        Defaults to the ``STRICT_ENV_CHECK`` environment variable (``true``/``1``).
    auto_install:
        Override the default behaviour for attempting to install missing
        dependencies. When ``True`` the checker will attempt ``pip install`` for
        missing packages, when ``False`` it will only report the missing
        dependencies. By default the value is sourced from the
        ``AUTO_INSTALL_DEPENDENCIES`` environment variable.

    Returns
    -------
    dict[str, Any]
        A diagnostic report containing missing dependencies, hardware
        information and manual pre-run checks.
    """

    logger = logging.getLogger(__name__)

    if strict is None:
        strict_env = os.getenv("STRICT_ENV_CHECK", "").strip().lower()
        strict = strict_env in {"1", "true", "yes"}

    check_results, manual_tests = _run_automated_preflight()

    python_check = next(
        (c for c in check_results if c.get("name") == "python-runtime"),
        None,
    )
    if python_check and (python_check.get("status") or "").lower() == "failed":
        raise EnvironmentCheckError(
            python_check.get("detail") or "Unsupported Python runtime detected.",
            manual_tests=manual_tests,
            checks=check_results,
        )
    missing_unique = _collect_missing_dependencies()
    install_attempts: dict[str, str] = {}
    auto_install_enabled = (
        AUTO_INSTALL_DEPENDENCIES_DEFAULT if auto_install is None else auto_install
    )

    if missing_unique:
        if auto_install_enabled:
            install_attempts = _attempt_dependency_install(missing_unique)
        else:
            install_attempts = {pkg: "auto-install disabled" for pkg in missing_unique}

        missing_unique = _collect_missing_dependencies()

        if find_spec("psutil") is not None and psutil is None:
            try:
                psutil_module = importlib.import_module("psutil")
            except Exception:  # pragma: no cover - rely on runtime error below
                psutil_module = None
            else:
                globals()["psutil"] = psutil_module

    if missing_unique:
        raise EnvironmentCheckError(
            "Missing dependencies: "
            + ", ".join(missing_unique)
            + ". Install with 'pip install -r requirements.txt' and re-run the check.",
            missing_dependencies=missing_unique,
            install_attempts=install_attempts,
            manual_tests=manual_tests,
            checks=check_results,
        )

    if psutil is None:
        raise EnvironmentCheckError(
            "psutil is required for environment diagnostics but could not be imported.",
            install_attempts=install_attempts,
            manual_tests=manual_tests,
            checks=check_results,
        )

    mem_info = psutil.virtual_memory()
    mem_gb = mem_info.total / 1_000_000_000
    cores = psutil.cpu_count() or 1
    hardware = {"memory_gb": mem_gb, "cpu_cores": cores}

    if mem_gb < MIN_RAM_GB or cores < MIN_CORES:
        message = (
            f"System resources too low ({mem_gb:.1f}GB RAM, {cores} cores). "
            f"Minimum required is {MIN_RAM_GB}GB RAM and {MIN_CORES} core."
        )
        if strict:
            raise EnvironmentCheckError(
                message,
                hardware=hardware,
                manual_tests=manual_tests,
                checks=check_results,
            )
        logger.warning(message)

    if mem_gb < REC_RAM_GB or cores < REC_CORES:
        logger.warning(
            "Running on low-spec hardware (%.1fGB RAM, %d cores); performance will be reduced.",
            mem_gb,
            cores,
        )
        _adjust_config_for_low_spec()

    return {
        "missing_dependencies": missing_unique,
        "install_attempts": install_attempts,
        "hardware": hardware,
        "automated_checks": check_results,
        "manual_tests": manual_tests,
    }


def _create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate MT5 environment prerequisites and render the manual checklist.",
    )
    parser.set_defaults(strict=None, auto_install=None)

    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        help="Fail when minimum hardware requirements are not met.",
    )
    strict_group.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Log hardware issues as warnings instead of raising.",
    )

    install_group = parser.add_mutually_exclusive_group()
    install_group.add_argument(
        "--auto-install",
        dest="auto_install",
        action="store_true",
        help="Attempt to install missing dependencies with pip.",
    )
    install_group.add_argument(
        "--no-auto-install",
        dest="auto_install",
        action="store_false",
        help="Only report missing dependencies without installing them.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output the diagnostics as JSON instead of formatted text.",
    )
    return parser


def _print_report(report: dict[str, Any]) -> None:
    missing = report.get("missing_dependencies") or []
    if missing:
        print("Missing dependencies:")
        for package in missing:
            print(f"  - {package}")

    install_attempts = report.get("install_attempts") or {}
    if install_attempts:
        print("Dependency auto-install attempts:")
        for package, status in install_attempts.items():
            print(f"  - {package}: {status}")

    hardware = report.get("hardware") or {}
    if hardware:
        memory_gb = hardware.get("memory_gb")
        cpu_cores = hardware.get("cpu_cores")
        if memory_gb is not None and cpu_cores is not None:
            print(
                "Hardware detected: "
                f"{memory_gb:.1f}GB RAM, {cpu_cores} cores"
            )

    checks = report.get("automated_checks") or []
    if checks:
        print("Automated pre-run checks:")
        for check in checks:
            name = check.get("name", "Check")
            status = check.get("status", "unknown")
            detail = check.get("detail")
            print(f"  - {name}: {status}")
            if detail:
                detail_str = detail if isinstance(detail, str) else str(detail)
                wrapped = detail_str.splitlines() or [detail_str]
                for line in wrapped:
                    print(f"      {line}")

    manual_tests = report.get("manual_tests") or []
    if manual_tests:
        print("Manual pre-run checklist:")
        for idx, item in enumerate(manual_tests, start=1):
            print(f"  {idx}. {item}")


def main(argv: list[str] | None = None) -> int:
    parser = _create_arg_parser()
    args = parser.parse_args(argv)

    try:
        report = ensure_environment(strict=args.strict, auto_install=args.auto_install)
    except EnvironmentCheckError as exc:
        payload = {
            "error": str(exc),
            "missing_dependencies": exc.missing_dependencies,
            "install_attempts": exc.install_attempts,
            "hardware": exc.hardware,
            "manual_tests": exc.manual_tests,
            "automated_checks": exc.check_results,
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            _print_report(payload)
            print(f"Environment check failed: {exc}")
        return 1

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
