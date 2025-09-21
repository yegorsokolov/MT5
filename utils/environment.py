import logging
import os
import pkgutil
import sys
from pathlib import Path
from typing import Any

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
    from config_models import AppConfig
except Exception:  # pragma: no cover - handled later
    AppConfig = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQ_FILE = PROJECT_ROOT / "requirements-core.txt"
CONFIG_FILE = Path(os.getenv("CONFIG_FILE", PROJECT_ROOT / "config.yaml"))

MIN_RAM_GB = 2
REC_RAM_GB = 8
MIN_CORES = 1
REC_CORES = 4

def _check_dependencies() -> list[str]:
    missing: list[str] = []
    if not REQ_FILE.exists():
        return missing
    for line in REQ_FILE.read_text().splitlines():
        pkg = line.strip()
        if not pkg or pkg.startswith("#"):
            continue
        pkg_name = pkg.split("==")[0]
        module_name = pkg_name.replace("-", "_")
        if pkgutil.find_loader(module_name) is None:
            missing.append(pkg_name)
    return missing


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


def ensure_environment() -> None:
    """Validate dependencies and warn about weak hardware.

    If resources are below recommended thresholds the configuration is adjusted
    for slower machines. If requirements are not met, a clear error is raised so
    the user can install the missing packages.
    """
    logger = logging.getLogger(__name__)

    missing = _check_dependencies()
    if psutil is None:
        missing.append("psutil")

    if missing:
        raise RuntimeError(
            "Missing dependencies: "
            + ", ".join(missing)
            + ". Install with 'pip install -r requirements-core.txt'"
            " or the appropriate extras."
        )

    mem_gb = psutil.virtual_memory().total / 1_000_000_000
    cores = psutil.cpu_count() or 1
    if mem_gb < MIN_RAM_GB or cores < MIN_CORES:
        raise RuntimeError(
            f"System resources too low ({mem_gb:.1f}GB RAM, {cores} cores). "
            f"Minimum required is {MIN_RAM_GB}GB RAM and {MIN_CORES} core."
        )

    if mem_gb < REC_RAM_GB or cores < REC_CORES:
        logger.warning(
            "Running on low-spec hardware (%.1fGB RAM, %d cores); "
            "performance will be reduced.",
            mem_gb,
            cores,
        )
        _adjust_config_for_low_spec()


if __name__ == "__main__":
    ensure_environment()
