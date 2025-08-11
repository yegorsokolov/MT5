"""Utilities to build symbol relation graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pickle
import networkx as nx
import pandas as pd


def build_correlation_graph(
    returns: pd.DataFrame, threshold: float = 0.5, path: Optional[Path] = None
) -> nx.Graph:
    """Build a graph where edges connect symbols with high rolling correlation.

    Parameters
    ----------
    returns:
        DataFrame indexed by time with columns per symbol containing return series.
    threshold:
        Minimum absolute correlation to create an edge.
    path:
        Optional path to persist the graph via :func:`networkx.write_gpickle`.
    """

    corr = returns.corr()
    g = nx.Graph()
    for sym in corr.columns:
        g.add_node(sym)
    for i, sym_i in enumerate(corr.columns):
        for j in range(i + 1, len(corr.columns)):
            sym_j = corr.columns[j]
            if abs(corr.iloc[i, j]) >= threshold:
                g.add_edge(sym_i, sym_j, weight=corr.iloc[i, j])
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(g, f)
    return g
