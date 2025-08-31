"""Build heterogeneous event-instrument graphs.

This module constructs a directed graph that links macroeconomic events to
financial instruments.  It ingests the output of :mod:`news.aggregator` and
:mod:`data.macro_features` modules.  Event nodes are connected to instrument
nodes when the instrument name contains the event's currency code.  Edge
weights reflect the qualitative importance of the event.

Graphs are persisted as pickled ``networkx`` graphs under ``data/graphs`` by
default.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

try:  # pragma: no cover - pandas is optional for typing only
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = Any  # type: ignore


_IMPORTANCE_WEIGHT = {"low": 0.5, "medium": 1.0, "high": 2.0}


def build_event_graph(
    events: List[Dict],
    macro_df: pd.DataFrame,
    path: Optional[Path] = Path("data/graphs/event_graph.pkl"),
) -> nx.DiGraph:
    """Construct a heterogeneous graph from events and macro series.

    Parameters
    ----------
    events:
        List of event dictionaries as produced by :class:`news.aggregator.NewsAggregator`.
    macro_df:
        Dataframe returned by :func:`data.macro_features.load_macro_series`.
    path:
        Optional path to persist the graph snapshot.  If ``None`` the graph is
        not written to disk.
    """

    g = nx.DiGraph()

    # Instrument nodes from macro features
    for col in macro_df.columns:
        if col == "Date":
            continue
        g.add_node(col, type="instrument")

    # Event nodes and edges
    for ev in events:
        ev_id = str(ev.get("id") or f"{ev.get('timestamp')}_{ev.get('event')}")
        attrs = {k: v for k, v in ev.items() if k != "id"}
        attrs.setdefault("type", "event")
        g.add_node(ev_id, **attrs)
        currency = str(ev.get("currency") or "").upper()
        if not currency:
            continue
        weight = _IMPORTANCE_WEIGHT.get(str(ev.get("importance", "")).lower(), 1.0)
        for node, data in g.nodes(data=True):
            if data.get("type") == "instrument" and currency in node.upper():
                g.add_edge(ev_id, node, weight=weight)

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(g, f)

    return g


__all__ = ["build_event_graph"]
