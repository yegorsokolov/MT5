from __future__ import annotations
"""Persistence of model performance across market regimes.

This module aggregates historical profit and loss for each model within a
market regime.  Daily values are computed from executed trades and weekly
values are derived from the sum of daily PnL within each ISO week.  Results
are stored in a Parquet file so other components such as the strategy router
can quickly bias decisions toward historically strong models for the current
regime.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_PATH = Path("analytics/regime_performance.parquet")
TRADES_PATH = Path("reports/trades.csv")


@dataclass
class RegimePerformanceStore:
    """Store and recompute per-regime model performance."""

    path: Path = DEFAULT_PATH

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _load(self) -> pd.DataFrame:
        if self.path.exists():
            df = pd.read_parquet(self.path)
            df["date"] = pd.to_datetime(df["date"])
            return df
        return pd.DataFrame(
            columns=["date", "model", "regime", "pnl_daily", "pnl_weekly"],
        )

    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Return a copy of persisted regime performance statistics."""

        return self._load().copy()

    # ------------------------------------------------------------------
    def recompute(self, trades: Optional[pd.DataFrame] = None) -> None:
        """Recompute regime performance from raw trade data.

        Parameters
        ----------
        trades:
            Optional dataframe with at least ``exit_time``, ``pnl``, ``model``
            and ``regime`` columns.  When omitted, ``reports/trades.csv`` is
            loaded if available.
        """

        if trades is None:
            if not TRADES_PATH.exists():
                return
            trades = pd.read_csv(TRADES_PATH, parse_dates=["exit_time"])

        if trades.empty:
            return

        df = trades.copy()
        if "exit_time" not in df:
            raise ValueError("trades dataframe missing 'exit_time'")
        if "pnl" not in df:
            raise ValueError("trades dataframe missing 'pnl'")
        if "model" not in df:
            raise ValueError("trades dataframe missing 'model'")
        if "regime" not in df:
            raise ValueError("trades dataframe missing 'regime'")

        df["date"] = pd.to_datetime(df["exit_time"]).dt.normalize()
        daily = (
            df.groupby(["date", "model", "regime"], as_index=False)["pnl"].sum()
        )
        daily.rename(columns={"pnl": "pnl_daily"}, inplace=True)

        daily["week"] = daily["date"].dt.to_period("W-MON").dt.start_time
        weekly = (
            daily.groupby(["week", "model", "regime"], as_index=False)[
                "pnl_daily"
            ].sum()
        )
        weekly.rename(columns={"pnl_daily": "pnl_weekly"}, inplace=True)

        result = daily.merge(weekly, on=["week", "model", "regime"], how="left")
        result.drop(columns=["week"], inplace=True)
        result.sort_values("date", inplace=True)
        result.to_parquet(self.path, index=False)

    # ------------------------------------------------------------------
    def get(self, regime: Optional[int | str] = None) -> pd.DataFrame:
        """Return performance records optionally filtered by regime."""

        df = self._load()
        if regime is not None:
            df = df[df["regime"] == regime]
        return df.copy()


__all__ = ["RegimePerformanceStore"]
