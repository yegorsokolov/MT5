from __future__ import annotations

"""Utilities for building per-symbol sentiment and impact vectors from cached headlines."""

from pathlib import Path
from typing import List, Dict

import pandas as pd

from . import stock_headlines


def load_vectors(window: int = 3, cache_dir: Path | None = None) -> pd.DataFrame:
    """Return per-symbol sentiment and impact vectors.

    Parameters
    ----------
    window:
        Number of past headlines to include per vector.  The most recent
        headline is index ``0``.
    cache_dir:
        Optional override for the headline cache directory used by
        :func:`news.stock_headlines.load_scores`.
    """

    scores = stock_headlines.load_scores(cache_dir=cache_dir)
    if scores.empty:
        cols = ["symbol", "timestamp"]
        cols += [f"news_sentiment_{i}" for i in range(window)]
        cols += [f"news_impact_{i}" for i in range(window)]
        return pd.DataFrame(columns=cols)

    scores = scores.sort_values("timestamp")
    out: List[Dict[str, object]] = []
    for sym, grp in scores.groupby("symbol"):
        grp = grp.reset_index(drop=True)
        sentiments = grp.get("sentiment", pd.Series([0.0] * len(grp)))
        impacts = grp.get("news_movement_score", pd.Series([0.0] * len(grp)))
        for i, row in grp.iterrows():
            s_vec = sentiments.iloc[max(0, i - window + 1) : i + 1].tolist()[::-1]
            s_vec = [0.0] * (window - len(s_vec)) + s_vec
            i_vec = impacts.iloc[max(0, i - window + 1) : i + 1].tolist()[::-1]
            i_vec = [0.0] * (window - len(i_vec)) + i_vec
            rec: Dict[str, object] = {
                "symbol": sym,
                "timestamp": row["timestamp"],
            }
            rec.update({f"news_sentiment_{j}": s_vec[j] for j in range(window)})
            rec.update({f"news_impact_{j}": i_vec[j] for j in range(window)})
            out.append(rec)
    return pd.DataFrame(out)


def top_headlines(symbol: str, n: int = 3, cache_dir: Path | None = None) -> List[Dict[str, object]]:
    """Return the ``n`` most recent headlines for ``symbol`` with scores."""

    scores = stock_headlines.load_scores(cache_dir=cache_dir)
    if scores.empty:
        return []
    df = scores[scores["symbol"] == symbol].sort_values("timestamp", ascending=False).head(n)
    records: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        records.append(
            {
                "title": row.get("title", ""),
                "sentiment": float(row.get("sentiment", 0.0)),
                "impact": float(row.get("news_movement_score", 0.0)),
                "url": row.get("url", ""),
            }
        )
    return records


__all__ = ["load_vectors", "top_headlines"]
