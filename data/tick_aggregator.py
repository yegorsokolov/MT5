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
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import pandas as pd

from analytics.metrics_store import record_metric

logger = logging.getLogger(__name__)


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
            divergence = float((mid_p - mid_s).abs().mean())
            record_metric("tick_source_divergence", divergence, tags={"symbol": symbol})

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


__all__ = ["TickAggregator", "init", "fetch_ticks"]
