from __future__ import annotations

"""Score recorded decision rationales against realised PnL."""

from pathlib import Path
from typing import Dict

import pandas as pd


def score_rationales(
    trade_path: str | Path = "logs/trade_history.parquet",
    report_dir: str | Path = "reports/rationale_scores",
) -> Dict[str, pd.DataFrame]:
    """Join decisions with PnL and compute quality metrics.

    Parameters
    ----------
    trade_path:
        Parquet file containing decision records with realised ``pnl``.
    report_dir:
        Directory where summary parquet files will be written.

    Returns
    -------
    dict
        Mapping of metric name to DataFrame/Series that were persisted.
    """

    trade_path = Path(trade_path)
    if not trade_path.exists():  # pragma: no cover - sanity check
        raise FileNotFoundError(trade_path)

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(trade_path)

    # Reason accuracy ---------------------------------------------------
    reason_accuracy = df.groupby("reason")["pnl"].apply(lambda x: (x > 0).mean())
    reason_accuracy.loc["__overall__"] = (df["pnl"] > 0).mean()
    reason_accuracy = reason_accuracy.rename("accuracy").to_frame()
    reason_accuracy.to_parquet(report_dir / "reason_accuracy.parquet")

    # Feature-importance drift -----------------------------------------
    drift_df = pd.DataFrame()
    if "feature_importance" in df.columns:
        fi = df["feature_importance"].dropna().apply(pd.Series)
        if not fi.empty:
            drift_df = fi.mean().rename("importance").to_frame()
            drift_df.to_parquet(report_dir / "feature_importance_drift.parquet")

    # Algorithm win rates -----------------------------------------------
    win_rates = df.groupby("algorithm")["pnl"].apply(lambda x: (x > 0).mean())
    win_rates = win_rates.rename("win_rate").to_frame()
    win_rates.to_parquet(report_dir / "algorithm_win_rates.parquet")

    return {
        "reason_accuracy": reason_accuracy,
        "feature_importance_drift": drift_df,
        "algorithm_win_rates": win_rates,
    }


def load_algorithm_win_rates(
    path: str | Path = "reports/rationale_scores/algorithm_win_rates.parquet",
) -> Dict[str, float]:
    """Load persisted algorithm win rates from ``path``.

    Parameters
    ----------
    path:
        Location of the parquet file.  If it does not exist an empty mapping is
        returned.
    """

    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    if "win_rate" in df.columns:
        df = df["win_rate"]
    return df.astype(float).to_dict()


__all__ = ["score_rationales", "load_algorithm_win_rates"]
