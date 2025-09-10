from __future__ import annotations

"""Train or tune the baseline strategy."""

import argparse
from pathlib import Path

import pandas as pd

from tuning.baseline_opt import backtest, run_search


def _load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline strategy")
    parser.add_argument(
        "--data", required=True, help="Historical data file (csv or parquet)"
    )
    parser.add_argument(
        "--tune", action="store_true", help="Run Optuna parameter search"
    )
    parser.add_argument(
        "--trials", type=int, default=30, help="Number of Optuna trials"
    )
    args = parser.parse_args()

    df = _load_data(args.data)
    if args.tune:
        best = run_search(df, n_trials=args.trials)
        print(best)
    else:
        params = {
            "short_window": 5,
            "long_window": 20,
            "atr_window": 14,
            "cvd_threshold": 0.0,
            "stop_mult": 3.0,
        }
        score = backtest(params, df)
        print(f"Sharpe: {score:.3f}")


if __name__ == "__main__":  # pragma: no cover - script entry
    import warnings

    warnings.warn(
        "train_baseline.py is deprecated; use 'python train_cli.py baseline' instead",
        DeprecationWarning,
    )
    main()
