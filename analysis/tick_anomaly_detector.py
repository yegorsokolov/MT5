"""Tick-level anomaly detection utilities.

This module implements a lightweight online z-score based detector
for streaming tick data.  It checks three conditions:

* bid/ask spreads deviating from historical mean
* non-monotonic timestamps
* large price jumps in the mid price

The detector maintains running statistics using Welford's algorithm so it
can operate on an incoming stream without storing historical ticks.
``filter`` returns a dataframe with anomalies removed along with the number of
anomalous rows filtered.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import math
import pandas as pd


@dataclass
class _RunningStat:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0


class TickAnomalyDetector:
    """Online z-score detector for tick data."""

    def __init__(self, threshold: float = 3.0) -> None:
        self.threshold = threshold
        self.spread_stat = _RunningStat()
        self.jump_stat = _RunningStat()
        self.last_ts: pd.Timestamp | None = None
        self.prev_mid: float | None = None
        self.total: int = 0
        self.anoms: int = 0

    # ------------------------------------------------------------------
    def filter(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Return dataframe with anomalies removed and count of anomalies."""
        if df.empty:
            return df, 0

        clean_rows = []
        for row in df.itertuples():
            self.total += 1
            ts = row.Timestamp
            bid = float(row.Bid)
            ask = float(row.Ask)
            spread = ask - bid
            mid = (ask + bid) / 2.0
            anomaly = False

            # Timestamp order
            if self.last_ts is not None and ts <= self.last_ts:
                anomaly = True

            # Spread z-score (and basic sanity check ask > bid)
            spread_std = self.spread_stat.std
            if ask <= bid:
                anomaly = True
            elif self.spread_stat.n > 10 and spread_std > 0 and abs(spread - self.spread_stat.mean) > self.threshold * spread_std:
                anomaly = True

            # Price jump z-score
            if self.prev_mid is not None:
                jump = mid - self.prev_mid
                jump_std = self.jump_stat.std
                if self.jump_stat.n > 10 and jump_std > 0 and abs(jump - self.jump_stat.mean) > self.threshold * jump_std:
                    anomaly = True
            else:
                jump = 0.0

            if anomaly:
                self.anoms += 1
                continue

            clean_rows.append((ts, bid, ask, row.BidVolume, row.AskVolume))
            # Update stats only with clean ticks
            self.last_ts = ts
            self.spread_stat.update(spread)
            self.jump_stat.update(jump)
            self.prev_mid = mid

        clean_df = pd.DataFrame(clean_rows, columns=df.columns)
        return clean_df, self.anoms


# Maintain detectors per symbol ------------------------------------------------
_detectors: Dict[str, TickAnomalyDetector] = {}


def filter(df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, int]:
    """Filter anomalies for a given symbol."""
    det = _detectors.setdefault(symbol, TickAnomalyDetector())
    before = det.anoms
    clean_df, _ = det.filter(df)
    return clean_df, det.anoms - before
