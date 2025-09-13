"""Multi-objective utilities for classification trading models.

This module computes trade metrics such as F1, expected return and
maximum drawdown and combines them via a configurable weighted sum.  The
implementation avoids heavy dependencies so it can be reused in tests
and lightweight optimisation loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
try:  # pragma: no cover - sklearn optional
    from sklearn.metrics import f1_score
except Exception:  # noqa: E722 - fallback implementation
    def f1_score(y_true, y_pred, zero_division=0):
        yt = np.asarray(list(y_true), dtype=int)
        yp = np.asarray(list(y_pred), dtype=int)
        tp = np.sum((yt == 1) & (yp == 1))
        fp = np.sum((yt == 0) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        if prec + rec == 0:
            return float(zero_division)
        return float(2 * prec * rec / (prec + rec))


@dataclass
class TradeMetrics:
    """Container holding metrics for a trading strategy."""

    f1: float
    expected_return: float
    drawdown: float

    def weighted_sum(self, weights: Mapping[str, float]) -> float:
        """Return weighted objective value.

        ``weights`` is a mapping containing weights for ``"f1"``,
        ``"return"`` and ``"drawdown"``.  The drawdown component is
        subtracted as it is a cost.
        """

        w_f1 = float(weights.get("f1", 0.0))
        w_ret = float(weights.get("return", 0.0))
        w_dd = float(weights.get("drawdown", 0.0))
        return w_f1 * self.f1 + w_ret * self.expected_return - w_dd * self.drawdown


def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> TradeMetrics:
    """Compute F1, expected return and drawdown from predictions.

    Parameters
    ----------
    y_true, y_pred:
        Iterable of true labels and binary predictions.
    """

    yt = np.asarray(list(y_true), dtype=int)
    yp = np.asarray(list(y_pred), dtype=int)
    f1 = f1_score(yt, yp, zero_division=0)
    returns = np.where(yp == 1, np.where(yt == 1, 1.0, -1.0), 0.0)
    expected = float(returns.mean()) if returns.size else 0.0
    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    dd = float(np.max(peak - cumulative)) if cumulative.size else 0.0
    return TradeMetrics(f1=f1, expected_return=expected, drawdown=dd)


def weighted_sum(metrics: TradeMetrics, weights: Mapping[str, float]) -> float:
    """Compute weighted sum from :class:`TradeMetrics` and ``weights``."""

    return metrics.weighted_sum(weights)


__all__ = ["TradeMetrics", "compute_metrics", "weighted_sum"]
