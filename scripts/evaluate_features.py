import copy
import csv
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    # Allow running via ``python scripts/evaluate_features.py`` without installing the package.
    sys.path.insert(0, str(REPO_ROOT))

import mt5.log_utils as log_utils
from mt5.log_utils import setup_logging, log_exceptions
from mt5.backtest import run_rolling_backtest
from utils import load_config, update_config

_LOGGING_INITIALIZED = False


def init_logging() -> logging.Logger:
    """Initialise structured logging for feature evaluation runs."""

    global _LOGGING_INITIALIZED
    if not _LOGGING_INITIALIZED:
        setup_logging()
        _LOGGING_INITIALIZED = True
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)

LOG_DIR = getattr(log_utils, "LOG_DIR", Path(__file__).resolve().parents[1] / "logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = LOG_DIR / "feature_eval.csv"


def _feature_flags(cfg: dict) -> list[str]:
    """Return config keys controlling optional features."""
    return [k for k in cfg.keys() if k.startswith("use_")]


@log_exceptions
def main() -> None:
    init_logging()
    cfg = load_config()
    flags = _feature_flags(cfg)
    base_metrics = run_rolling_backtest(cfg)
    base_sharpe = base_metrics.get("avg_sharpe", float("nan"))
    logger.info("Baseline avg sharpe %s", base_sharpe)

    records = []

    for flag in flags:
        enabled_cfg = copy.deepcopy(cfg)
        enabled_cfg[flag] = True
        disabled_cfg = copy.deepcopy(cfg)
        disabled_cfg[flag] = False

        if not cfg.get(flag, False):
            update_config(flag, True, "enable experimental feature")
            cfg[flag] = True

        metrics_enabled = run_rolling_backtest(enabled_cfg)
        metrics_disabled = run_rolling_backtest(disabled_cfg)

        sharpe_on = metrics_enabled.get("avg_sharpe", float("nan"))
        sharpe_off = metrics_disabled.get("avg_sharpe", float("nan"))

        records.append({
            "feature": flag,
            "enabled_sharpe": sharpe_on,
            "disabled_sharpe": sharpe_off,
        })

        if sharpe_off > sharpe_on:
            reason = f"{flag} reduced sharpe from {sharpe_on:.4f} to {sharpe_off:.4f}"
            update_config(flag, False, reason)
            logger.info("Disabling %s", flag)
        else:
            reason = f"{flag} beneficial with sharpe {sharpe_on:.4f}"
            update_config(flag, True, reason)
            logger.info("Keeping %s enabled", flag)

    with open(_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "enabled_sharpe", "disabled_sharpe"])
        if f.tell() == 0:
            writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
