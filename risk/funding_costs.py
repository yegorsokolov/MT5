"""Fetch broker funding costs and margin rules for instruments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class FundingInfo:
    """Funding and margin information for a trading instrument.

    Parameters
    ----------
    swap_rate : float
        Expected daily funding cost as a fraction of position notional.
    margin_requirement : float
        Fraction of notional that must be posted as margin.
    available_margin : float
        Account-level margin currently available for new positions.
    """

    swap_rate: float = 0.0
    margin_requirement: float = 0.1
    available_margin: float = float("inf")


# Default values used for tests and offline operation.  In production these
# would be retrieved from the broker's API.
_DEFAULT_SWAP_RATES: Dict[str, float] = {"EURUSD": 0.0001}
_DEFAULT_MARGIN_RULES: Dict[str, float] = {"EURUSD": 0.02}
_AVAILABLE_MARGIN: float = 1_000_000.0


def fetch_funding_info(symbol: str) -> FundingInfo:
    """Return funding info for ``symbol``.

    This stub implementation uses static sample data so tests can run without
    external broker dependencies.
    """

    swap = _DEFAULT_SWAP_RATES.get(symbol, 0.0)
    margin_req = _DEFAULT_MARGIN_RULES.get(symbol, 0.1)
    return FundingInfo(
        swap_rate=swap, margin_requirement=margin_req, available_margin=_AVAILABLE_MARGIN
    )
