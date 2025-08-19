from __future__ import annotations

"""Estimate expected value of entering a trade now versus waiting.

This module provides a light‑weight scorer that combines simple order‑book
signals with recent volatility and regime embeddings.  The intent is not to be a
perfect forecast but rather to offer a quick heuristic that can be evaluated in
unit tests without heavy dependencies.  The scorer returns an expected value
metric after deducting an optional transaction cost.

A small utility function is also provided to log predicted vs. realised values
so the accuracy of the heuristic can be monitored offline.
"""

from dataclasses import dataclass
from pathlib import Path
import os
import csv
from typing import Sequence

import numpy as np


@dataclass
class EntryValueScorer:
    """Heuristic expected value scorer.

    Parameters
    ----------
    depth_weight:
        Coefficient applied to the order book depth imbalance.
    vol_weight:
        Penalty applied to recent volatility.
    cost:
        Fixed transaction cost deducted from the score.  A positive score after
        costs indicates entering immediately is preferable to waiting.
    """

    depth_weight: float = 0.5
    vol_weight: float = 0.3
    cost: float = 0.0

    def score(
        self,
        depth_imbalance: float,
        volatility: float,
        regime_embed: Sequence[float] | None = None,
    ) -> float:
        """Return expected value of entering now minus waiting.

        The model is intentionally simple: depth imbalance provides a positive
        contribution while volatility adds a penalty.  Regime embeddings, when
        supplied, are squashed through ``tanh`` and averaged to obtain a single
        adjustment factor.
        """

        emb = np.asarray(regime_embed if regime_embed is not None else [], dtype=float)
        regime_adj = float(np.tanh(emb).mean()) if emb.size else 0.0
        expected = self.depth_weight * float(depth_imbalance) - self.vol_weight * float(volatility)
        expected += regime_adj
        return expected - self.cost


def log_entry_value(timestamp: str, symbol: str, predicted: float, realised: float | None = None) -> None:
    """Append predicted vs realised entry values to ``reports/entry_value``.

    The output directory can be overridden with the ``ENTRY_VALUE_REPORT_PATH``
    environment variable to ease testing.
    """

    report_dir = Path(
        os.getenv(
            "ENTRY_VALUE_REPORT_PATH",
            Path(__file__).resolve().parents[1] / "reports" / "entry_value",
        )
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    file = report_dir / "entry_value.csv"
    write_header = not file.exists()
    with file.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Timestamp", "Symbol", "predicted", "realised"])
        writer.writerow([timestamp, symbol, predicted, "" if realised is None else realised])


__all__ = ["EntryValueScorer", "log_entry_value"]
