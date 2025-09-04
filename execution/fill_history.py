from __future__ import annotations

"""Utilities for recording execution fill details."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List
import csv

HISTORY_FILE = Path("fill_history.csv")


@dataclass
class FillRecord:
    """Record of an executed child order."""

    timestamp: datetime
    slippage: float
    latency: float
    depth: float


def record_fill(*, slippage: float, latency: float, depth: float, file: Path = HISTORY_FILE) -> None:
    """Append a fill record to ``file``.

    Parameters
    ----------
    slippage: float
        Realized slippage in basis points.
    latency: float
        Time taken to execute the order in seconds.
    depth: float
        Available book depth at the time of the order.
    file: :class:`pathlib.Path`, optional
        Location of the history CSV file.
    """
    file.parent.mkdir(parents=True, exist_ok=True)
    exists = file.exists()
    with file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "slippage", "latency", "depth"])
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "slippage": slippage,
                "latency": latency,
                "depth": depth,
            }
        )


def load_history(file: Path = HISTORY_FILE) -> List[FillRecord]:
    """Load fill history records from ``file``."""
    records: List[FillRecord] = []
    if not file.exists():
        return records
    with file.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                records.append(
                    FillRecord(
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        slippage=float(row["slippage"]),
                        latency=float(row["latency"]),
                        depth=float(row["depth"]),
                    )
                )
            except Exception:
                continue
    return records
