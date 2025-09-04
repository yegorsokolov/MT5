from __future__ import annotations

"""Transaction cost analysis for broker fills.

This module records the time between order placement and fill as well as the
realised slippage for each broker. Metrics are persisted via
:mod:`analytics.metrics_store` which allows the dashboard and other
components to surface per-broker performance.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from analytics.metrics_store import record_metric


@dataclass
class BrokerTCA:
    """Record per-broker latency and slippage."""

    records: List[Dict[str, object]] = field(default_factory=list)

    def record(
        self,
        broker: str,
        order_ts: pd.Timestamp,
        fill_ts: pd.Timestamp,
        slippage_bps: float,
    ) -> None:
        """Record a fill event for ``broker``.

        Parameters
        ----------
        broker:
            Broker identifier.
        order_ts:
            Timestamp when the order was sent.
        fill_ts:
            Timestamp when the order was filled.
        slippage_bps:
            Realised slippage in basis points.
        """

        latency_ms = (fill_ts - order_ts).total_seconds() * 1000.0
        self.records.append(
            {
                "broker": broker,
                "order_ts": order_ts,
                "fill_ts": fill_ts,
                "latency_ms": latency_ms,
                "slippage_bps": slippage_bps,
            }
        )
        # Persist individual observations to the metrics store so that other
        # components (e.g. dashboards) can access the data.
        try:
            record_metric("broker_fill_latency_ms", latency_ms, {"broker": broker})
            record_metric("broker_slippage_bps", slippage_bps, {"broker": broker})
        except Exception:
            # Metrics logging is best-effort – failures should not impact trading
            pass

    def dataframe(self) -> pd.DataFrame:
        """Return recorded observations as a dataframe."""

        if not self.records:
            return pd.DataFrame(
                columns=["broker", "order_ts", "fill_ts", "latency_ms", "slippage_bps"]
            )
        return pd.DataFrame(self.records)


# Global profiler instance used by execution modules
broker_tca = BrokerTCA()
