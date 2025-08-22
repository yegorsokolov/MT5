from __future__ import annotations

"""Track aggregate long and short notional exposure."""

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict

from analytics.metrics_store import record_metric


@dataclass
class NetExposure:
    """Maintain long/short exposures and enforce portfolio limits."""

    max_long: float = float("inf")
    max_short: float = float("inf")
    long: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    short: Dict[str, float] = field(default_factory=lambda: defaultdict(float))

    # ------------------------------------------------------------------
    def _totals(self) -> tuple[float, float]:
        return sum(self.long.values()), sum(self.short.values())

    # ------------------------------------------------------------------
    def limit(self, symbol: str, notional: float) -> float:
        """Return notional allowed for a proposed trade."""

        long_tot, short_tot = self._totals()
        if notional > 0:
            available = self.max_long - long_tot
            if available <= 0:
                return 0.0
            return min(notional, available)
        elif notional < 0:
            available = self.max_short - short_tot
            if available <= 0:
                return 0.0
            return max(notional, -available)
        return 0.0

    # ------------------------------------------------------------------
    def update(self, symbol: str, notional: float) -> None:
        """Record executed trade notional for ``symbol``."""

        if notional > 0:
            self.long[symbol] += notional
        elif notional < 0:
            self.short[symbol] += -notional
        long_tot, short_tot = self._totals()
        try:
            record_metric("long_exposure", long_tot)
            record_metric("short_exposure", short_tot)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def totals(self) -> Dict[str, float]:
        """Return dictionary of long, short and net exposure."""

        long_tot, short_tot = self._totals()
        return {"long": long_tot, "short": short_tot, "net": long_tot - short_tot}
