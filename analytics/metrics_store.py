from __future__ import annotations

"""Persistence layer for daily performance metrics.

This module stores daily aggregates such as return, Sharpe ratio and
maximum drawdown.  Metrics are persisted to a Parquet file which can be
loaded as a time–series for analysis or reporting.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

DEFAULT_PATH = Path("analytics/metrics.parquet")
# Separate store for generic time-series metrics
TS_PATH = Path("analytics/metrics_timeseries.parquet")


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


# ---------------------------------------------------------------------------
def _ts_load(path: Path = TS_PATH) -> pd.DataFrame:
    """Load the time-series metrics dataframe."""

    if path.exists():
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    return pd.DataFrame(columns=["timestamp", "name", "value"])


def record_metric(name: str, value: float, tags: Optional[Dict[str, Any]] = None, *, path: Path = TS_PATH) -> None:
    """Persist a single metric observation.

    Parameters
    ----------
    name:
        Metric name.
    value:
        Metric value.  Stored as ``float``.
    tags:
        Optional dictionary of additional columns to persist.
    path:
        Destination Parquet file.  Defaults to ``TS_PATH``.
    """

    tags = tags or {}
    row: Dict[str, Any] = {
        "timestamp": pd.Timestamp.utcnow(),
        "name": name,
        "value": float(value),
    }
    row.update(tags)
    df = _ts_load(path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def model_cache_hit() -> None:
    """Record a cache hit for a loaded model."""
    record_metric("model_cache_hits", 1.0)


def model_unload() -> None:
    """Record that a cached model was unloaded."""
    record_metric("model_unloads", 1.0)


def query_metrics(
    name: str | None = None,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    tags: Optional[Dict[str, Any]] = None,
    path: Path = TS_PATH,
) -> pd.DataFrame:
    """Return metrics from the time-series store.

    Filters by name, time range and optional tags.  The returned dataframe
    always includes ``timestamp``, ``name`` and ``value`` columns.
    """

    df = _ts_load(path)
    if name:
        df = df[df["name"] == name]
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end)]
    if tags:
        for k, v in tags.items():
            if k in df.columns:
                df = df[df[k] == v]
            else:
                # No matching tag column implies no rows
                df = df.iloc[0:0]
                break
    return df.reset_index(drop=True)

