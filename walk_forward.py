from __future__ import annotations

from pathlib import Path
import pandas as pd

from utils import load_config
from backtest import run_rolling_backtest
from log_utils import setup_logging, log_exceptions

logger = setup_logging()

_LOG_PATH = Path(__file__).resolve().parent / "logs" / "walk_forward.csv"
_LOG_PATH.parent.mkdir(exist_ok=True)


def aggregate_results(results: dict[str, dict]) -> pd.DataFrame:
    """Convert results dictionary to a DataFrame."""
    records = []
    for symbol, metrics in results.items():
        records.append({
            "symbol": symbol,
            "avg_sharpe": metrics.get("avg_sharpe"),
            "worst_drawdown": metrics.get("worst_drawdown"),
        })
    return pd.DataFrame.from_records(records, columns=["symbol", "avg_sharpe", "worst_drawdown"])


@log_exceptions
def main() -> pd.DataFrame | None:
    """Run rolling backtests for all configured symbols and log summary."""
    cfg = load_config()
    symbols = cfg.get("symbols") or [cfg.get("symbol")]

    results: dict[str, dict] = {}
    for sym in symbols:
        cfg_sym = dict(cfg)
        cfg_sym["symbol"] = sym
        logger.info("Running rolling backtest for %s", sym)
        metrics = run_rolling_backtest(cfg_sym)
        if metrics:
            results[sym] = metrics

    if not results:
        print("No backtest results")
        return None

    df = aggregate_results(results)
    header = not _LOG_PATH.exists()
    df.to_csv(_LOG_PATH, mode="a", header=header, index=False)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
