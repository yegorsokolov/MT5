import logging
import os
import re
import sys
from pathlib import Path
from typing import Any
from importlib.util import find_spec

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - handled later
    psutil = None  # type: ignore

try:
    from . import load_config_data, save_config
except Exception:  # pragma: no cover - handled later
    load_config_data = None  # type: ignore
    save_config = None  # type: ignore

try:
    from mt5.config_models import AppConfig
except Exception:  # pragma: no cover - handled later
    AppConfig = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQ_FILE = PROJECT_ROOT / "requirements-core.txt"
CONFIG_FILE = Path(os.getenv("CONFIG_FILE", PROJECT_ROOT / "config.yaml"))

MIN_RAM_GB = 2
REC_RAM_GB = 8
MIN_CORES = 1
REC_CORES = 4

_SPECIFIER_SPLIT_RE = re.compile(r"\s*(?:==|!=|<=|>=|~=|===|<|>|=)")


def _parse_requirement_line(line: str) -> tuple[str, str] | None:
    """Extract the requirement and canonical module name from a line.

    Returns ``None`` when the line does not represent an installable package
    (empty strings, comments, options, etc.).
    """

    if not line:
        return None

    candidate = line.split("#", 1)[0].strip()
    if not candidate:
        return None

    if candidate.startswith("#"):
        return None

    candidate = candidate.split(";", 1)[0].strip()
    if not candidate:
        return None

    if candidate.startswith("-"):
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


def _check_dependencies() -> list[str]:
    missing: list[str] = []
    if not REQ_FILE.exists():
        return missing

    for line in REQ_FILE.read_text().splitlines():
        parsed = _parse_requirement_line(line)
        if parsed is None:
            continue
        pkg_name, module_name = parsed
        if find_spec(module_name) is None:
            missing.append(pkg_name)
    return missing


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


def _render_manual_tests() -> list[str]:
    """Return instructions for manual pre-flight tests."""

    manual_tests = [
        "Launch the MT5 terminal and verify the bot can log in using your broker credentials.",
        "From the running bot, execute a simple ping to the MT5 terminal to confirm connectivity.",
        "Validate that Git operations (clone/pull/push) succeed using the configured credentials.",
        "Check that environment variables from the .env file are loaded before starting the bot.",
        "Exercise the oracle scalper pipeline (e.g. collect() then assess_probabilities()) to confirm external market data APIs respond.",
        "Start the inference FastAPI service (services.inference_server) and hit the /health endpoint to verify REST API integrations.",
        "Spin up the feature worker FastAPI app and ensure background tasks can subscribe to the message bus/broker queue.",
    ]

    if not _mt5_terminal_downloaded():
        manual_tests.insert(
            0,
            "Download and install the MetaTrader 5 terminal, then set MT5_TERMINAL_PATH or place it in the 'mt5/' folder.",
        )

    if not _detect_git_credentials():
        manual_tests.append(
            "Configure Git remotes and ensure access tokens/SSH keys are available for GitHub operations.",
        )

    if not _has_env_file():
        manual_tests.append(
            "Create a .env file with API keys, broker credentials, and database URIs as required by your deployment.",
        )

    return manual_tests


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


def ensure_environment(strict: bool | None = None) -> dict[str, Any]:
    """Validate dependencies, hardware and pre-run requirements.

    Parameters
    ----------
    strict:
        When ``True`` the function raises on hardware shortfalls. When
        ``False`` the function logs warnings for hardware checks. This does not
        affect dependency validation – required packages are always enforced.
        Defaults to the ``STRICT_ENV_CHECK`` environment variable (``true``/``1``).

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

    missing = _check_dependencies()
    if psutil is None:
        missing.append("psutil")

    missing_unique = sorted(set(missing))

    if missing_unique:
        raise RuntimeError(
            "Missing dependencies: "
            + ", ".join(missing_unique)
            + ". Install with 'pip install -r requirements-core.txt' or the appropriate extras, then re-run the check."
        )

    manual_tests = _render_manual_tests()

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
            raise RuntimeError(message)
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
        "hardware": hardware,
        "manual_tests": manual_tests,
    }


if __name__ == "__main__":
    report = ensure_environment()
    if report["missing_dependencies"]:
        print("Install missing dependencies:", ", ".join(report["missing_dependencies"]))
    if report.get("hardware"):
        hardware = report["hardware"]
        print(
            "Hardware detected: "
            f"{hardware['memory_gb']:.1f}GB RAM, {hardware['cpu_cores']} cores"
        )
    print("Manual pre-run checklist:")
    for idx, item in enumerate(report["manual_tests"], start=1):
        print(f"  {idx}. {item}")
