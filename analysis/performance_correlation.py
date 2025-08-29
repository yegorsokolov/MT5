"""Compute correlations between features and PnL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


@dataclass
class PerformanceCorrelation:
    """Utility to compute feature/PnL correlations.

    Parameters
    ----------
    features : Sequence[str]
        Feature columns to correlate against PnL.
    """

    features: Sequence[str]

    # ------------------------------------------------------------------
    def compute(self, df: pd.DataFrame, pnl: Iterable[float]) -> pd.DataFrame:
        """Return correlations between configured ``features`` and ``pnl``.

        The returned dataframe contains ``feature``, ``pearson`` and
        ``spearman`` columns regardless of the ``method`` setting.  Missing
        correlations (e.g. due to constant values) are represented by ``NaN``.
        """

        data = pd.DataFrame(df, copy=False)
        data = data[list(self.features)].reset_index(drop=True)
        data["pnl"] = list(pnl)
        rows = []
        for col in self.features:
            s = data[col]
            try:
                pearson = float(s.corr(data["pnl"], method="pearson"))
            except Exception:
                pearson = float("nan")
            try:
                spearman = float(s.corr(data["pnl"], method="spearman"))
            except Exception:
                spearman = float("nan")
            rows.append({"feature": col, "pearson": pearson, "spearman": spearman})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def compute_correlations(
    df: pd.DataFrame, pnl: Iterable[float], features: Sequence[str] | None = None
) -> pd.DataFrame:
    """Convenience wrapper returning feature/PnL correlations.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe.
    pnl : Iterable[float]
        Per-period PnL series aligned with ``df``.
    features : Sequence[str], optional
        Subset of columns from ``df`` to correlate.  If ``None`` all columns are
        used.
    """

    feats = list(features) if features is not None else list(df.columns)
    corr = PerformanceCorrelation(features=feats)
    return corr.compute(df[feats], pnl)


__all__ = ["PerformanceCorrelation", "compute_correlations"]
