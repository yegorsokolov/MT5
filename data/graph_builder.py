"""Utilities to build symbol relation graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pickle
from typing import Dict, Iterable, List

import networkx as nx
import numpy as np
import pandas as pd


def _find_price_column(df: pd.DataFrame) -> str:
    """Return the most likely price column in ``df``.

    The function searches for common column names such as ``Close`` or ``Bid``.
    If none are found the first numeric column is returned.  This keeps the
    helper lightweight so it can operate on the minimal datasets used in the
    unit tests.
    """

    for candidate in ["Close", "close", "Bid", "bid", "Price", "price"]:
        if candidate in df.columns:
            return candidate
    # fall back to the first numeric column
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            return col
    raise ValueError("No price column found")


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


def build_rolling_adjacency(
    df: pd.DataFrame,
    window: int = 30,
    method: str = "correlation",
    symbols: Iterable[str] | None = None,
    path: Optional[Path] = None,
) -> Dict[pd.Timestamp, np.ndarray]:
    """Compute rolling adjacency matrices between symbols.

    Parameters
    ----------
    df:
        Long-form DataFrame with columns ``Timestamp`` and ``Symbol`` and a
        price column (``Close``/``Bid``/``Price``).
    window:
        Rolling window size in observations.
    method:
        Either ``"correlation"`` or ``"cointegration"``.
    symbols:
        Optional iterable of symbols to include.  If ``None`` all symbols in
        ``df`` are used.
    path:
        Optional path to persist the dictionary of adjacency matrices using
        :func:`pickle.dump`.

    Returns
    -------
    Dict[pd.Timestamp, np.ndarray]
        Mapping from window end timestamp to ``(n_symbols, n_symbols)`` adjacency
        matrix.
    """

    if symbols is None:
        symbols = sorted(df["Symbol"].unique())
    else:
        symbols = list(symbols)

    price_col = _find_price_column(df)
    wide = (
        df.pivot(index="Timestamp", columns="Symbol", values=price_col)
        .sort_index()
        .ffill()
    )
    wide = wide[symbols]
    returns = wide.pct_change().dropna()
    matrices: Dict[pd.Timestamp, np.ndarray] = {}
    for end in returns.index[window - 1 :]:
        window_df = returns.loc[:end].tail(window)
        if method == "correlation":
            mat = window_df.corr().fillna(0.0).to_numpy()
        elif method == "cointegration":
            try:
                from statsmodels.tsa.stattools import coint  # type: ignore

                n = len(symbols)
                mat = np.zeros((n, n), dtype=float)
                for i in range(n):
                    for j in range(i + 1, n):
                        _, p, _ = coint(window_df.iloc[:, i], window_df.iloc[:, j])
                        mat[i, j] = mat[j, i] = float(p < 0.05)
            except Exception:  # pragma: no cover - optional dependency
                mat = np.eye(len(symbols))
        else:  # pragma: no cover - invalid method should fail
            raise ValueError("Unknown method")
        matrices[end] = mat

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(matrices, f)
    return matrices


__all__ = ["build_correlation_graph", "build_rolling_adjacency"]
