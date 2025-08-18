from __future__ import annotations

"""Extreme value theory utilities for tail-risk estimation.

This module provides helper functions to fit a Generalised Pareto
Distribution (GPD) to the tail of a return series and estimate the
probability of drawdowns beyond a given threshold.  Results can be logged
for later inspection under ``reports/tail_risk``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import genpareto


@dataclass
class EVTResult:
    """Parameters estimated for the tail distribution."""

    shape: float
    scale: float
    threshold: float
    tail_prob: float
    n_exceed: int


def estimate_tail_probability(
    returns: Sequence[float],
    threshold: float,
    level: float | None = None,
) -> Tuple[float, EVTResult]:
    """Estimate probability of a drawdown beyond ``level`` using GPD.

    Parameters
    ----------
    returns:
        Sequence of PnL or return observations.  Losses should be negative.
    threshold:
        Threshold (negative) used to fit the tail model.  Observations more
        severe than this level are used as exceedances.
    level:
        Target drawdown level (negative).  If omitted, ``threshold`` is used.

    Returns
    -------
    tuple
        ``(probability, EVTResult)`` where ``probability`` is the estimated
        probability that a new observation exceeds ``level``.
    """

    arr = np.asarray(returns, dtype=float)
    total = arr.size
    losses = -arr[arr < 0]
    u = abs(float(threshold))
    exceedances = losses[losses > u] - u
    n_exceed = exceedances.size
    if n_exceed < 3 or total == 0:
        result = EVTResult(0.0, 0.0, u, 0.0, int(n_exceed))
        return 0.0, result

    shape, loc, scale = genpareto.fit(exceedances, floc=0)
    result = EVTResult(float(shape), float(scale), float(u), 0.0, int(n_exceed))

    x = abs(level) if level is not None else u
    if x < u:
        x = u
    prob = (n_exceed / total) * genpareto.sf(x - u, shape, loc=0, scale=scale)
    result.tail_prob = float(prob)
    return result.tail_prob, result


def log_evt_result(
    result: EVTResult, *, breach: bool, out_dir: str = "reports/tail_risk"
) -> Path:
    """Append EVT parameters and tail probability to a CSV log.

    Parameters
    ----------
    result:
        :class:`EVTResult` returned by :func:`estimate_tail_probability`.
    breach:
        Whether the calculated tail probability exceeded a risk limit.
    out_dir:
        Destination directory for the log file.  Created if missing.
    """

    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    log_file = path / "evt_log.csv"
    df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp.utcnow(),
                "shape": result.shape,
                "scale": result.scale,
                "threshold": result.threshold,
                "tail_prob": result.tail_prob,
                "n_exceed": result.n_exceed,
                "breach": bool(breach),
            }
        ]
    )
    df.to_csv(log_file, mode="a", header=not log_file.exists(), index=False)
    return log_file


__all__ = [
    "EVTResult",
    "estimate_tail_probability",
    "log_evt_result",
]
