import logging
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

try:
    import psutil  # type: ignore
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "psutil"], check=False)
    import psutil  # type: ignore

try:
    import yaml  # type: ignore
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "pyyaml"], check=False)
    import yaml  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQ_FILE = PROJECT_ROOT / "requirements.txt"
CONFIG_FILE = Path(os.getenv("CONFIG_FILE", PROJECT_ROOT / "config.yaml"))

MIN_RAM_GB = 2
REC_RAM_GB = 8
MIN_CORES = 1
REC_CORES = 4


def _install_packages(packages: list[str]) -> None:
    if not packages:
        return
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    subprocess.run(cmd, check=False)


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


def _adjust_config_for_low_spec() -> None:
    if not CONFIG_FILE.exists():
        return
    with CONFIG_FILE.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    changed = False
    if cfg.get("batch_size", 64) > 32:
        cfg["batch_size"] = 32
        changed = True
    if cfg.get("n_jobs", 2) > 1:
        cfg["n_jobs"] = 1
        changed = True
    if changed:
        with CONFIG_FILE.open("w") as f:
            yaml.safe_dump(cfg, f)


def ensure_environment() -> None:
    """Install missing dependencies and warn about weak hardware.

    If resources are below recommended thresholds the configuration is adjusted
    for slower machines. If requirements are not met, a warning is logged so the
    user can upgrade the VM.
    """
    logger = logging.getLogger("environment")

    missing = _check_dependencies()
    if missing:
        logger.info("Installing missing packages: %s", ", ".join(missing))
        _install_packages(missing)
        remaining = _check_dependencies()
        if remaining:
            logger.error("Unable to install packages: %s", ", ".join(remaining))
            raise RuntimeError(
                "Missing dependencies: " + ", ".join(remaining)
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
