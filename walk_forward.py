from __future__ import annotations

from pathlib import Path
import logging
import pandas as pd

from utils import load_config
from backtest import run_rolling_backtest
from backtesting.walk_forward import rolling_windows
from log_utils import setup_logging, log_exceptions

_LOGGING_INITIALIZED = False


def init_logging() -> logging.Logger:
    """Initialise structured logging for walk-forward utilities."""

    global _LOGGING_INITIALIZED
    if not _LOGGING_INITIALIZED:
        setup_logging()
        _LOGGING_INITIALIZED = True
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)

# default location for walk forward summary output
_LOG_PATH = Path(__file__).resolve().parent / "logs" / "walk_forward_summary.csv"
_LOG_PATH.parent.mkdir(exist_ok=True)


def aggregate_results(results: dict[str, dict]) -> pd.DataFrame:
    """Convert a ``symbol -> metrics`` mapping into a DataFrame."""

    records = []
    for symbol, metrics in results.items():
        records.append(
            {
                "symbol": symbol,
                "avg_sharpe": metrics.get("avg_sharpe"),
                "worst_drawdown": metrics.get("worst_drawdown"),
            }
        )

    return pd.DataFrame.from_records(
        records, columns=["symbol", "avg_sharpe", "worst_drawdown"]
    )


def walk_forward_train(
    data: Path,
    window_length: int,
    step_size: int,
    model_type: str = "mean",
) -> pd.DataFrame:
    """Run a simple walk-forward training loop.

    Parameters
    ----------
    data:
        Path to a CSV or parquet file containing a ``return`` column.
    window_length:
        Number of rows to use for the training window.
    step_size:
        Size of the forward evaluation window and stride between windows.
    model_type:
        Which toy model to train.  Currently only ``"mean"`` is supported.

    Returns
    -------
    DataFrame
        Metrics for each window including the train/test split positions.
    """

    if data.suffix == ".csv":
        df = pd.read_csv(data)
    else:
        df = pd.read_parquet(data)

    windows = rolling_windows(df, window_length, step_size, step_size)
    records = []

    # import mlflow lazily so tests can stub it easily
    import mlflow

    for i, (train, test) in enumerate(windows):
        with mlflow.start_run(run_name=f"window_{i}"):
            if model_type == "mean":
                pred = train["return"].mean()
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            rmse = float(((test["return"] - pred) ** 2).mean() ** 0.5)
            mlflow.log_metric("rmse", rmse)

            records.append(
                {
                    "window": i,
                    "train_end": int(train.index.max()),
                    "test_start": int(test.index.min()),
                    "rmse": rmse,
                }
            )

    return pd.DataFrame.from_records(records)


@log_exceptions
def main() -> pd.DataFrame | None:
    init_logging()
    """Run rolling backtests for all configured symbols and log summary."""
    cfg = load_config()

    # ``walk_forward_configs`` allows supplying explicit configuration
    # dictionaries for each symbol.  If not provided we simply iterate over
    # the symbol list in the base config.
    cfgs = cfg.get("walk_forward_configs")
    if cfgs:
        configs = cfgs
    else:
        symbols = cfg.get("symbols") or [cfg.get("symbol")]
        configs = []
        for sym in symbols:
            cfg_sym = dict(cfg)
            cfg_sym["symbol"] = sym
            configs.append(cfg_sym)

    results: dict[str, dict] = {}
    for cfg_sym in configs:
        sym = cfg_sym.get("symbol")
        logger.info("Running rolling backtest for %s", sym)
        metrics = run_rolling_backtest(cfg_sym)
        if metrics:
            results[sym] = metrics

    if not results:
        logger.info("No backtest results")
        return None

    df = aggregate_results(results)
    header = not _LOG_PATH.exists()
    df.to_csv(_LOG_PATH, mode="a", header=header, index=False)
    logger.info("%s", df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
