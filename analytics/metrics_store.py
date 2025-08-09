from __future__ import annotations

"""Persistence layer for daily performance metrics.

This module stores daily aggregates such as return, Sharpe ratio and
maximum drawdown.  Metrics are persisted to a Parquet file which can be
loaded as a time–series for analysis or reporting.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_PATH = Path("analytics/metrics.parquet")


@dataclass
class MetricsStore:
    """Store and retrieve daily performance metrics.

    Parameters
    ----------
    path:
        Location of the Parquet file.  The directory is created if it does
        not already exist.
    """

    path: Path = DEFAULT_PATH

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _load(self) -> pd.DataFrame:
        if self.path.exists():
            df = pd.read_parquet(self.path)
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date")
        return pd.DataFrame(columns=["return", "sharpe", "drawdown", "regime"])

    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Return all persisted metrics as a dataframe."""
        return self._load().copy()

    # ------------------------------------------------------------------
    def append(
        self,
        date: pd.Timestamp,
        *,
        ret: float,
        sharpe: float,
        drawdown: float,
        regime: Optional[int | str] = None,
    ) -> None:
        """Append a new day's metrics and persist to disk."""

        df = self._load()
        date = pd.Timestamp(date).normalize()
        df.loc[date] = {
            "return": ret,
            "sharpe": sharpe,
            "drawdown": drawdown,
            "regime": regime,
        }
        df.sort_index(inplace=True)
        df.reset_index().to_parquet(self.path, index=False)

    # ------------------------------------------------------------------
    def get(self, start: Optional[str | pd.Timestamp] = None, end: Optional[str | pd.Timestamp] = None) -> pd.DataFrame:
        """Return metrics for a given date range."""

        df = self._load()
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        return df.copy()
