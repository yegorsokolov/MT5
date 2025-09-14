from __future__ import annotations

"""Utilities for recording live tick data to a partitioned Parquet store."""

import asyncio
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.data_quality import check_recency


class LiveRecorder:
    """Asynchronously append tick data frames to a Parquet dataset.

    Parameters
    ----------
    root: Path | str, optional
        Base directory for the dataset.  Partitions are created below this
        directory using ``date=YYYY-MM-DD`` folders.
    batch_size: int, optional
        Number of rows to buffer before writing a Parquet batch.
    flush_interval: float, optional
        Seconds to wait before flushing a partial batch.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        batch_size: int = 500,
        flush_interval: float = 1.0,
    ) -> None:
        self.root = Path(root) if root is not None else Path("data") / "live"
        self.root.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.latencies: list[float] = []

    async def run(
        self,
        in_queue: asyncio.Queue[pd.DataFrame],
        out_queue: asyncio.Queue[pd.DataFrame],
        *,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """Continuously consume ticks and persist them to Parquet.

        Parameters
        ----------
        in_queue:
            Queue yielding tick data frames.
        out_queue:
            Queue receiving batches after they have been written.
        stop_event:
            Optional event used in tests to request shutdown once *in_queue* is
            empty.
        """

        buffer: list[pd.DataFrame] = []
        while True:
            if stop_event and stop_event.is_set() and in_queue.empty():
                break
            try:
                df = await asyncio.wait_for(in_queue.get(), timeout=self.flush_interval)
            except asyncio.TimeoutError:
                df = None
            if df is not None:
                in_queue.task_done()
                buffer.append(df)
                if sum(len(b) for b in buffer) < self.batch_size:
                    continue
            if not buffer:
                continue
            batch = pd.concat(buffer, ignore_index=True)
            start = time.perf_counter()
            await asyncio.to_thread(self._write_batch, batch)
            self.latencies.append(time.perf_counter() - start)
            await out_queue.put(batch)
            buffer.clear()
        if buffer:
            batch = pd.concat(buffer, ignore_index=True)
            start = time.perf_counter()
            await asyncio.to_thread(self._write_batch, batch)
            self.latencies.append(time.perf_counter() - start)
            await out_queue.put(batch)
        await out_queue.put(None)

    # ------------------------------------------------------------------
    def _write_batch(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        frame = frame.copy()
        if not pd.api.types.is_datetime64_any_dtype(frame["Timestamp"]):
            frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], utc=True)
        # Log if data is excessively stale to avoid backfilling with old ticks
        check_recency(frame, max_age="365d")
        frame.sort_values("Timestamp", inplace=True)
        frame["date"] = frame["Timestamp"].dt.strftime("%Y-%m-%d")
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_to_dataset(table, root_path=str(self.root), partition_cols=["date"])

    def avg_latency(self) -> float:
        """Return the mean write latency for recorded batches."""
        return (
            float(sum(self.latencies) / len(self.latencies)) if self.latencies else 0.0
        )


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
