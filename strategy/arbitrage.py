from __future__ import annotations

"""Basic cross-broker arbitrage detection utilities."""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd

from data.tick_aggregator import compute_spread_matrix
from .router import StrategyRouter

logger = logging.getLogger(__name__)

LOG_DIR = Path("reports/arbitrage")


@dataclass
class ArbitrageSignal:
    """Simple representation of an arbitrage opportunity."""

    symbol: str
    broker_a: str
    broker_b: str
    spread: float
    timestamp: pd.Timestamp


def _log_signal(sig: ArbitrageSignal) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / "signals.csv"
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "symbol", "broker_a", "broker_b", "spread"])
        writer.writerow(
            [sig.timestamp.isoformat(), sig.symbol, sig.broker_a, sig.broker_b, f"{sig.spread:.6f}"]
        )


def detect_arbitrage(
    symbol: str,
    quotes: Dict[str, Tuple[float, float]],
    threshold: float,
    router: StrategyRouter,
) -> List[ArbitrageSignal]:
    """Detect cross-broker spreads and publish signals via ``router``.

    Parameters
    ----------
    symbol:
        Instrument identifier.
    quotes:
        Mapping of broker name to ``(bid, ask)`` tuples.
    threshold:
        Minimum absolute spread required to trigger a signal.
    router:
        Instance of :class:`StrategyRouter` used to publish the signal.
    """

    matrix = compute_spread_matrix(quotes)
    signals: List[ArbitrageSignal] = []
    if matrix.empty:
        return signals

    ts = pd.Timestamp.utcnow()
    brokers = list(matrix.index)
    for i in range(len(brokers)):
        for j in range(i + 1, len(brokers)):
            a = brokers[i]
            b = brokers[j]
            spread = float(matrix.loc[a, b])
            if spread > threshold:
                sig = ArbitrageSignal(symbol, a, b, spread, ts)
                signals.append(sig)
                _log_signal(sig)
                try:
                    router.select({"arbitrage": spread})
                except Exception:  # pragma: no cover - best effort
                    logger.exception("Failed to route arbitrage signal")
    return signals


__all__ = ["ArbitrageSignal", "detect_arbitrage"]
