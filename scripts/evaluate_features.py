import copy
from pathlib import Path
import csv
import logging

from log_utils import setup_logging, log_exceptions
from utils import load_config, update_config
from backtest import run_rolling_backtest

setup_logging()
logger = logging.getLogger(__name__)

_LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "feature_eval.csv"
_LOG_FILE.parent.mkdir(exist_ok=True)


def _feature_flags(cfg: dict) -> list[str]:
    """Return config keys controlling optional features."""
    return [k for k in cfg.keys() if k.startswith("use_")]


@log_exceptions
def main() -> None:
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
