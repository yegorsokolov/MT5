from __future__ import annotations

"""Aggregate tick data from multiple broker feeds.

The aggregator concurrently fetches ticks from a primary and secondary
broker, aligns them on timestamp and resolves any conflicting ticks using a
latency preference.  Basic health information about each source is tracked via
latency measurements and divergence between the two feeds which are recorded in
``analytics.metrics_store``.

The public :func:`fetch_ticks` function exposes the unified tick stream.  It
falls back to the healthiest source when one of the brokers disconnects or
fails to provide data.
"""

import asyncio
import logging
import time
import os
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, List

import pandas as pd

from analytics.metrics_store import record_metric

try:  # pragma: no cover - alerting optional in tests
    from utils.alerting import send_alert
except Exception:  # pragma: no cover - fallback stub
    def send_alert(msg: str) -> None:  # type: ignore
        logger = logging.getLogger(__name__)
        logger.warning("ALERT: %s", msg)

_DIVERGENCE_THRESHOLD = float(os.getenv("BROKER_SPREAD_THRESHOLD", "0.5"))
_LOG_PATH = Path("logs/broker_anomalies.csv")

logger = logging.getLogger(__name__)


@dataclass
class DivergenceEvent:
    symbol: str
    divergence: float
    timestamp: pd.Timestamp
    resolved: bool = False


_subscribers: List[asyncio.Queue[DivergenceEvent]] = []
_diverged = False


def divergence_alerts() -> asyncio.Queue[DivergenceEvent]:
    """Return a queue that receives broker divergence events."""
    q: asyncio.Queue[DivergenceEvent] = asyncio.Queue()
    _subscribers.append(q)
    return q


def _publish(event: DivergenceEvent) -> None:
    for q in list(_subscribers):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:  # pragma: no cover - unlikely
            continue


def _log_divergence(symbol: str, div: float, ts: pd.Timestamp) -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = _LOG_PATH.exists()
    with _LOG_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "symbol", "divergence"])
        writer.writerow([ts.isoformat(), symbol, f"{div:.6f}"])


def _convert_ticks(ticks: Any) -> pd.DataFrame:
    """Convert raw tick structures to the standard dataframe format."""
    if not ticks:
        return pd.DataFrame()
    df = pd.DataFrame(ticks)
    if df.empty:
        return df
    df["Timestamp"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"bid": "Bid", "ask": "Ask", "volume": "Volume"}, inplace=True)
    df = df[["Timestamp", "Bid", "Ask", "Volume"]]
    df["BidVolume"] = df["Volume"]
    df["AskVolume"] = df["Volume"]
    df.drop(columns=["Volume"], inplace=True)
    return df


def compute_spread_matrix(prices: dict[str, Any]) -> pd.DataFrame:
    """Return pairwise absolute mid-price deltas between brokers.

    Parameters
    ----------
    prices:
        Mapping of broker name to either a ``(bid, ask)`` tuple, a single
        numeric mid price or a dataframe containing ``Bid`` and ``Ask``
        columns.  Only the most recent row of dataframes is considered.

    Returns
    -------
    pandas.DataFrame
        Square matrix whose ``i, j`` entry contains ``|mid_i - mid_j|`` for the
        corresponding brokers.  Missing brokers or empty inputs result in an
        empty dataframe.
    """

    if not prices:
        return pd.DataFrame()

    mids: dict[str, float] = {}
    for name, value in prices.items():
        bid: float
        ask: float
        if isinstance(value, pd.DataFrame):
            if value.empty:
                continue
            row = value.iloc[-1]
            bid = float(row.get("Bid", row.get("bid", 0.0)))
            ask = float(row.get("Ask", row.get("ask", bid)))
        else:
            try:
                bid, ask = value  # type: ignore[misc]
            except Exception:
                bid = ask = float(value)  # treat as mid price
        mids[name] = (bid + ask) / 2.0

    if not mids:
        return pd.DataFrame()

    brokers = list(mids)
    data = []
    for a in brokers:
        row: list[float] = []
        for b in brokers:
            row.append(abs(mids[a] - mids[b]))
        data.append(row)
    return pd.DataFrame(data, index=brokers, columns=brokers)


@dataclass
class TickAggregator:
    """Aggregate ticks from two broker sources."""

    primary: Any
    secondary: Any

    async def _fetch_from(self, source: Any, symbol: str, n: int) -> Tuple[str, pd.DataFrame, float]:
        """Fetch ticks from ``source`` measuring latency."""
        name = getattr(source, "__name__", source.__class__.__name__)
        start = time.perf_counter()
        try:
            ticks = await asyncio.to_thread(
                source.copy_ticks_from, symbol, int(time.time()) - n, n, source.COPY_TICKS_ALL
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to fetch ticks from %s", name)
            ticks = None
        latency = time.perf_counter() - start
        record_metric("tick_source_latency", latency, tags={"source": name})
        df = _convert_ticks(ticks) if ticks is not None else pd.DataFrame()
        return name, df, latency

    async def fetch(self, symbol: str, n: int = 1000) -> pd.DataFrame:
        """Return unified ticks for ``symbol`` from the healthiest source."""
        prim_name, prim_df, prim_lat = await self._fetch_from(self.primary, symbol, n)
        sec_name, sec_df, sec_lat = await self._fetch_from(self.secondary, symbol, n)

        if prim_df.empty and sec_df.empty:
            return pd.DataFrame()
        if prim_df.empty:
            return sec_df
        if sec_df.empty:
            return prim_df

        # Both sources returned data. Compute divergence on overlapping timestamps
        overlap = pd.merge(
            prim_df, sec_df, on="Timestamp", suffixes=("_p", "_s"), how="inner"
        )
        if not overlap.empty:
            mid_p = (overlap["Bid_p"] + overlap["Ask_p"]) / 2
            mid_s = (overlap["Bid_s"] + overlap["Ask_s"]) / 2
            spread = (mid_p - mid_s).abs()
            divergence = float(spread.mean())
            record_metric("tick_source_divergence", divergence, tags={"symbol": symbol})
            max_div = float(spread.max())
            ts_max = overlap.loc[spread.idxmax(), "Timestamp"]
            global _diverged
            if max_div > _DIVERGENCE_THRESHOLD:
                send_alert(
                    f"Broker price divergence {max_div:.6f} on {symbol}"
                )
                _log_divergence(symbol, max_div, ts_max)
                _publish(DivergenceEvent(symbol, max_div, ts_max, False))
                _diverged = True
            elif _diverged:
                _publish(DivergenceEvent(symbol, 0.0, ts_max, True))
                _diverged = False

        # Prefer ticks from the lower latency source for overlapping timestamps
        if prim_lat <= sec_lat:
            chosen, other = prim_df, sec_df
        else:
            chosen, other = sec_df, prim_df

        other_unique = other[~other["Timestamp"].isin(chosen["Timestamp"])]
        combined = pd.concat([chosen, other_unique], ignore_index=True)
        combined.sort_values("Timestamp", inplace=True)
        combined.reset_index(drop=True, inplace=True)
        return combined


_aggregator: Optional[TickAggregator] = None


def init(primary: Any, secondary: Any) -> None:
    """Initialise the global tick aggregator."""
    global _aggregator
    _aggregator = TickAggregator(primary, secondary)


async def fetch_ticks(symbol: str, n: int = 1000) -> pd.DataFrame:
    """Fetch unified ticks using the global :class:`TickAggregator` instance."""
    if _aggregator is None:
        raise RuntimeError("Tick aggregator not initialized")
    return await _aggregator.fetch(symbol, n)


__all__ = [
    "TickAggregator",
    "init",
    "fetch_ticks",
    "divergence_alerts",
    "DivergenceEvent",
    "compute_spread_matrix",
]
