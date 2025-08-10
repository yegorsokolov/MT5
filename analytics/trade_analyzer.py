from __future__ import annotations

"""Tools for analyzing trade performance after execution."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any, Dict

import pandas as pd
import json


@dataclass
class TradeAnalyzer:
    """Compute aggregate statistics from executed trades.

    Parameters
    ----------
    trades:
        ``pandas.DataFrame`` with at minimum the columns ``entry_time``,
        ``exit_time``, ``entry_price``, ``exit_price``, ``volume`` and ``pnl``.
    """

    trades: pd.DataFrame

    # ------------------------------------------------------------------
    @classmethod
    def from_records(cls, records: Iterable[Dict[str, Any]]) -> "TradeAnalyzer":
        """Create an instance from an iterable of trade dictionaries."""

        df = pd.DataFrame(list(records))
        if not df.empty:
            df["entry_time"] = pd.to_datetime(df["entry_time"])
            df["exit_time"] = pd.to_datetime(df["exit_time"])
        return cls(df)

    # ------------------------------------------------------------------
    def average_hold_time(self) -> pd.Timedelta:
        """Return the mean holding period for all trades."""

        if self.trades.empty:
            return pd.Timedelta(0)
        durations = self.trades["exit_time"] - self.trades["entry_time"]
        return durations.mean()

    # ------------------------------------------------------------------
    def pnl_by_duration(self) -> Dict[int, float]:
        """Aggregate PnL by trade duration in minutes.

        Returns a dictionary mapping integer duration (in minutes) to the
        corresponding total PnL for trades of that length.
        """

        if self.trades.empty:
            return {}
        df = self.trades.copy()
        df["duration_min"] = (
            (df["exit_time"] - df["entry_time"]).dt.total_seconds() // 60
        ).astype(int)
        grouped = df.groupby("duration_min")["pnl"].sum()
        return {int(k): float(v) for k, v in grouped.items()}

    # ------------------------------------------------------------------
    def turnover(self) -> float:
        """Return total traded notional across all trades."""

        if self.trades.empty:
            return 0.0
        df = self.trades
        notional_entry = (df["entry_price"] * df["volume"]).abs()
        notional_exit = (df["exit_price"] * df["volume"]).abs()
        return float((notional_entry + notional_exit).sum())

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """Return a dictionary with key statistics."""

        avg_hold = self.average_hold_time()
        return {
            "average_hold_time": avg_hold.total_seconds(),
            "pnl_by_duration": self.pnl_by_duration(),
            "turnover": self.turnover(),
        }

    # ------------------------------------------------------------------
    def save_report(self, path: str | Path) -> Path:
        """Persist summary statistics as JSON and return the path."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.summary(), fh)
        return path


__all__ = ["TradeAnalyzer"]
