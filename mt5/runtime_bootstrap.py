"""Runtime bootstrap helpers for automated bot start-up.

This module consolidates the ad-hoc shell helpers that operators previously ran
manually (such as ``scripts/run_bot.sh``) so that executing ``python -m mt5`` is
enough to provision background services, verify the MetaTrader bridge and launch
auxiliary tooling.  Each helper degrades gracefully when optional dependencies
or operating-system specific features are unavailable and can be toggled through
environment variables:

``AUTO_BOOTSTRAP_MT5``
    Set to ``0`` to disable all automatic bootstrap logic.
``AUTO_LAUNCH_MT5``
    Controls whether the MetaTrader terminal is started automatically
    (defaults to ``1``).
``SKIP_TERMINAL_CHECK``
    When ``1`` the connectivity check performed via ``setup_terminal.py`` is
    skipped.
``INSTALL_HEARTBEAT_SCRIPT``
    Controls whether the ConnectionHeartbeat script is copied into the terminal
    after a successful connectivity check (defaults to ``1``).
``START_DASHBOARD``
    Mirrors the legacy shell helper. When not ``0`` the Streamlit dashboard is
    launched automatically if Streamlit is installed.
``START_ARTIFACT_SYNC``
    Launch the background artifact synchronisation loop (defaults to ``1``).
``SYNC_INTERVAL_SECONDS``
    Interval forwarded to the artifact synchronisation helper.

The helpers intentionally import the historical scripts on-demand so the logic
stays centralised even though the scripts remain executable from the command
line.  Import errors are logged as warnings and never abort the bootstrap
sequence.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

_BOOTSTRAP_COMPLETED = False
_TERMINAL_PROCESS: Optional[subprocess.Popen[str]] = None
_DASHBOARD_PROCESS: Optional[subprocess.Popen[str]] = None
_DASHBOARD_LOG_HANDLE: Optional[Any] = None
_ARTIFACT_THREAD_STARTED = False


def _load_script_module(name: str) -> ModuleType | None:
    """Import a helper from :mod:`scripts` without requiring a package."""

    path = SCRIPTS_DIR / f"{name}.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"mt5_scripts.{name}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib guard
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - best effort logging
        logging.getLogger(__name__).warning("Failed to import %s: %s", path, exc)
        return None
    return module


def _discover_terminal_path(logger: logging.Logger) -> Path | None:
    """Populate ``MT5_TERMINAL_PATH`` using the detection helper when available."""

    env_override = os.getenv("MT5_TERMINAL_PATH")
    if env_override:
        candidate = Path(env_override).expanduser()
        if candidate.exists():
            return candidate

    module = _load_script_module("detect_mt5_terminal")
    if module is None:
        fallback = Path(env_override or "/opt/mt5")
        os.environ.setdefault("MT5_TERMINAL_PATH", str(fallback))
        return fallback

    discover = getattr(module, "_discover_terminal", None)
    fallback_fn = getattr(module, "_fallback_terminal_dir", None)

    terminal_dir: Path | None = None
    if callable(discover):
        try:
            result = discover()
        except Exception as exc:  # pragma: no cover - discovery is best effort
            logger.warning("MetaTrader 5 auto-discovery failed: %s", exc)
        else:
            if result is not None:
                terminal_dir = Path(result)

    if terminal_dir is None and callable(fallback_fn):
        try:
            terminal_dir = Path(fallback_fn())
        except Exception as exc:  # pragma: no cover - fallback should succeed
            logger.warning("MetaTrader 5 fallback resolution failed: %s", exc)

    if terminal_dir is None:
        terminal_dir = Path(env_override or "/opt/mt5")

    os.environ.setdefault("MT5_TERMINAL_PATH", str(terminal_dir))

    write_env = getattr(module, "_write_env_file", None)
    env_file = getattr(module, "DEFAULT_ENV_FILE", None)
    if callable(write_env) and isinstance(env_file, (str, Path)):
        env_path = Path(env_file)
        try:
            write_env(env_path, str(terminal_dir))
        except Exception as exc:  # pragma: no cover - permissions best effort
            logger.warning("Failed to persist MT5_TERMINAL_PATH to %s: %s", env_path, exc)
        else:
            logger.info("MetaTrader 5 terminal path set to %s", terminal_dir)

    return terminal_dir


def _resolve_terminal_executable(path: Path) -> Path | None:
    if path.is_file():
        return path
    for name in ("terminal64.exe", "terminal.exe", "terminal"):
        candidate = path / name
        if candidate.exists():
            return candidate
    return None


def _launch_terminal(logger: logging.Logger, terminal_dir: Path | None) -> bool:
    """Start the MetaTrader terminal if possible and requested."""

    if os.getenv("AUTO_LAUNCH_MT5", "1") == "0":
        return False
    if sys.platform.startswith("win"):
        logger.debug("Skipping automatic MetaTrader launch on Windows hosts")
        return False
    if terminal_dir is None:
        logger.debug("Terminal directory unknown; unable to auto-launch MetaTrader")
        return False

    executable = _resolve_terminal_executable(terminal_dir)
    if executable is None:
        logger.warning("Could not locate a MetaTrader terminal inside %s", terminal_dir)
        return False

    global _TERMINAL_PROCESS
    if _TERMINAL_PROCESS is not None and _TERMINAL_PROCESS.poll() is None:
        logger.debug("MetaTrader terminal already running (PID %s)", _TERMINAL_PROCESS.pid)
        return False

    command: list[str] = []
    xvfb = shutil.which("xvfb-run")
    if xvfb and not os.environ.get("DISPLAY"):
        command.extend([xvfb, "-d"])

    wine = shutil.which("wine")
    if wine:
        command.extend([wine, str(executable)])
    else:
        if executable.suffix.lower() == ".exe":
            logger.warning(
                "Wine is not available; cannot auto-start Windows binary %s",
                executable,
            )
            return False
        command.append(str(executable))

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(executable.parent),
        )
    except Exception as exc:  # pragma: no cover - launch depends on system
        logger.warning("Failed to launch MetaTrader terminal: %s", exc)
        return False

    _TERMINAL_PROCESS = process
    logger.info("Launched MetaTrader terminal via: %s", " ".join(command))
    return True


def _parse_login(logger: logging.Logger) -> tuple[Optional[int], Optional[str], Optional[str]]:
    raw_login = os.getenv("MT5_LOGIN")
    login: Optional[int] = None
    if raw_login:
        try:
            login = int(raw_login)
        except ValueError:
            logger.warning("Ignoring MT5_LOGIN=%s (not an integer)", raw_login)
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    return login, password, server


def _verify_terminal(logger: logging.Logger, terminal_dir: Path | None) -> bool:
    if os.getenv("SKIP_TERMINAL_CHECK", "0") == "1":
        logger.debug("Skipping MetaTrader connectivity check (SKIP_TERMINAL_CHECK=1)")
        return False

    module = _load_script_module("setup_terminal")
    if module is None:
        logger.debug("setup_terminal helper unavailable; skipping connectivity check")
        return False

    attempt = getattr(module, "attempt_connection", None)
    resolve = getattr(module, "_resolve_terminal_path", None)
    default_dir_fn = getattr(module, "_default_mt5_dir", None)
    copy_heartbeat = getattr(module, "_copy_heartbeat_script", None)

    if terminal_dir is None and callable(default_dir_fn):
        try:
            terminal_dir = Path(default_dir_fn())
        except Exception as exc:  # pragma: no cover - fallback should succeed
            logger.warning("Failed to determine MetaTrader directory: %s", exc)
            terminal_dir = None

    terminal_path: Optional[Path] = None
    if callable(resolve) and terminal_dir is not None:
        try:
            terminal_path = resolve(Path(terminal_dir))
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.warning("Unable to resolve MetaTrader terminal executable: %s", exc)

    login, password, server = _parse_login(logger)

    kwargs = {
        "terminal": terminal_path if terminal_path and terminal_path.exists() else None,
        "login": login,
        "password": password,
        "server": server,
    }

    success = False
    if callable(attempt):
        try:
            result = attempt(**kwargs)
        except Exception as exc:  # pragma: no cover - depends on MT5 runtime
            logger.warning("MetaTrader connectivity check raised an exception: %s", exc)
        else:
            success = bool(getattr(result, "success", False))
            if success:
                login_id = getattr(result, "account_login", None)
                broker = getattr(result, "broker", None)
                logger.info(
                    "MetaTrader bridge verified%s%s",
                    f" (login {login_id})" if login_id else "",
                    f" via {broker}" if broker else "",
                )
            else:
                error = getattr(result, "error", "unknown error")
                logger.warning("MetaTrader bridge verification failed: %s", error)

            if success and os.getenv("INSTALL_HEARTBEAT_SCRIPT", "1") != "0" and callable(copy_heartbeat):
                try:
                    heartbeat = copy_heartbeat(terminal_dir or Path("."))
                except Exception as exc:  # pragma: no cover - filesystem best effort
                    logger.warning("Failed to install heartbeat script: %s", exc)
                else:
                    if heartbeat:
                        logger.info("Heartbeat script installed at %s", heartbeat)

    return success


def _start_dashboard(logger: logging.Logger) -> None:
    if os.getenv("START_DASHBOARD", "1") == "0":
        logger.debug("START_DASHBOARD=0; skipping Streamlit dashboard launch")
        return

    streamlit = shutil.which("streamlit")
    if not streamlit:
        logger.debug("Streamlit not available; dashboard will not be launched")
        return

    dashboard_script = PROJECT_ROOT / "webui" / "dashboard.py"
    if not dashboard_script.exists():
        logger.debug("Dashboard script %s missing; skipping launch", dashboard_script)
        return

    global _DASHBOARD_PROCESS, _DASHBOARD_LOG_HANDLE
    if _DASHBOARD_PROCESS is not None and _DASHBOARD_PROCESS.poll() is None:
        logger.debug("Streamlit dashboard already running (PID %s)", _DASHBOARD_PROCESS.pid)
        return

    try:
        from mt5.log_utils import LOG_DIR
    except Exception:  # pragma: no cover - log_utils optional during tests
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
    else:
        log_dir = LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "dashboard.log"
    try:
        log_file = open(log_path, "a", encoding="utf-8")
    except OSError as exc:
        logger.warning("Unable to open dashboard log %s: %s", log_path, exc)
        log_file = None

    port = os.getenv("DASHBOARD_PORT", "8501")
    command = [
        streamlit,
        "run",
        str(dashboard_script),
        "--server.port",
        str(port),
        "--server.address",
        "0.0.0.0",
    ]

    try:
        process = subprocess.Popen(
            command,
            stdout=log_file or subprocess.DEVNULL,
            stderr=log_file or subprocess.DEVNULL,
            cwd=str(PROJECT_ROOT),
        )
    except Exception as exc:  # pragma: no cover - depends on Streamlit install
        if log_file:
            log_file.close()
        logger.warning("Failed to launch Streamlit dashboard: %s", exc)
        return

    _DASHBOARD_PROCESS = process
    _DASHBOARD_LOG_HANDLE = log_file
    logger.info("Streamlit dashboard started on port %s", port)


def _start_artifact_sync(logger: logging.Logger) -> None:
    if os.getenv("START_ARTIFACT_SYNC", "1") == "0":
        logger.debug("START_ARTIFACT_SYNC=0; skipping artifact synchronisation thread")
        return

    global _ARTIFACT_THREAD_STARTED
    if _ARTIFACT_THREAD_STARTED:
        return

    module = _load_script_module("hourly_artifact_push")
    main_fn: Optional[Callable[[], None]] = getattr(module, "main", None) if module else None
    if not callable(main_fn):
        logger.debug("Artifact synchronisation helper unavailable; skipping thread")
        return

    thread = threading.Thread(target=main_fn, name="artifact-sync", daemon=True)
    thread.start()
    _ARTIFACT_THREAD_STARTED = True
    logger.info(
        "Background artifact synchronisation thread started (interval %ss)",
        os.getenv("SYNC_INTERVAL_SECONDS", "3600"),
    )


def ensure_runtime_bootstrap(force: bool = False) -> None:
    """Ensure runtime helpers are active before starting realtime components."""

    global _BOOTSTRAP_COMPLETED
    if _BOOTSTRAP_COMPLETED and not force:
        return
    if os.getenv("AUTO_BOOTSTRAP_MT5", "1") == "0":
        _BOOTSTRAP_COMPLETED = True
        return

    logger = logging.getLogger(__name__)

    terminal_dir: Path | None = None
    try:
        terminal_dir = _discover_terminal_path(logger)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Terminal discovery raised an exception: %s", exc)

    started = False
    try:
        started = _launch_terminal(logger, terminal_dir)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Automatic terminal launch failed: %s", exc)

    if started:
        grace = os.getenv("MT5_LAUNCH_GRACE", "5")
        try:
            delay = float(grace)
        except ValueError:
            delay = 5.0
        if delay > 0:
            time.sleep(delay)

    try:
        _verify_terminal(logger, terminal_dir)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("MetaTrader verification failed: %s", exc)

    try:
        _start_dashboard(logger)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Dashboard launch failed: %s", exc)

    try:
        _start_artifact_sync(logger)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Artifact synchronisation thread failed to start: %s", exc)

    _BOOTSTRAP_COMPLETED = True


__all__ = ["ensure_runtime_bootstrap"]

