from __future__ import annotations

"""Utilities for recording live tick data to a partitioned Parquet store."""

from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.data_quality import check_recency


class LiveRecorder:
    """Append tick data frames to a time partitioned Parquet dataset.

    Parameters
    ----------
    root: Path | str, optional
        Base directory for the dataset.  Partitions are created below this
        directory using ``date=YYYY-MM-DD`` folders.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else Path("data") / "live"
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def record(self, df: pd.DataFrame) -> None:
        """Persist *df* to the Parquet dataset.

        The dataframe must contain a ``Timestamp`` column convertible to
        ``datetime64[ns]``.  Each call writes a new file under the partition
        corresponding to the UTC date of the timestamp.
        """

        if df.empty:
            return

        frame = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(frame["Timestamp"]):
            frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], utc=True)
        # Log if data is excessively stale to avoid backfilling with old ticks
        check_recency(frame, max_age="365d")
        frame.sort_values("Timestamp", inplace=True)
        frame["date"] = frame["Timestamp"].dt.strftime("%Y-%m-%d")
        date_val = frame["date"].iloc[0]
        ts_val = frame["Timestamp"].iloc[0].strftime("%H%M%S")
        table = pa.Table.from_pandas(frame.drop(columns=["date"]))
        out_dir = self.root / f"date={date_val}"
        out_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_dir / f"ticks_{ts_val}.parquet")


def load_ticks(root: Path | str, since: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Load recorded ticks from *root* optionally filtering by timestamp."""

    path = Path(root)
    files = sorted(path.glob("**/*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    if since is not None:
        df = df[df["Timestamp"] > since]
    return df.sort_values("Timestamp").reset_index(drop=True)


__all__ = ["LiveRecorder", "load_ticks"]
