from __future__ import annotations

"""Utilities for currency-adjusted exposure tracking."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import pandas as pd
import requests


@dataclass
class CurrencyExposure:
    """Convert instrument exposures to a common base currency.

    Parameters
    ----------
    base_currency:
        Account base currency. All exposures are expressed in this currency.
    instrument_currencies:
        Mapping from instrument symbol to its quote currency.
    """

    base_currency: str = "USD"
    instrument_currencies: Dict[str, str] = field(default_factory=dict)
    rates: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def _fetch_rate(self, currency: str) -> float:
        """Return FX rate from ``currency`` to ``base_currency``."""

        if currency == self.base_currency:
            return 1.0
        if currency not in self.rates:
            try:
                resp = requests.get(
                    "https://api.exchangerate.host/latest",
                    params={"base": currency, "symbols": self.base_currency},
                    timeout=10,
                )
                data = resp.json()
                rate = float(data.get("rates", {}).get(self.base_currency, 1.0))
                self.rates[currency] = rate
            except Exception:
                self.rates[currency] = 1.0
        return self.rates[currency]

    # ------------------------------------------------------------------
    def convert(self, symbol: str | None, exposure: float) -> float:
        """Convert ``exposure`` for ``symbol`` into ``base_currency``."""

        if symbol is None:
            return exposure
        currency = self.instrument_currencies.get(symbol, self.base_currency)
        rate = self._fetch_rate(currency)
        return exposure * rate

    # ------------------------------------------------------------------
    def snapshot(
        self, exposures: Dict[str, float], path: str = "reports/currency_exposure"
    ) -> None:
        """Persist current exposures to ``path``.

        A CSV timestamped snapshot is saved along with a ``latest.json`` file for the
        dashboard.
        """

        try:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
            df = pd.DataFrame(
                {"symbol": list(exposures), "exposure": list(exposures.values())}
            )
            df.to_csv(p / f"{ts}.csv", index=False)
            df.set_index("symbol")["exposure"].to_json(p / "latest.json")
        except Exception:
            pass
