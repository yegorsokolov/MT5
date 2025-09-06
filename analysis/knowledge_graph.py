"""Construct a simple financial knowledge graph.

This module builds a heterogeneous knowledge graph linking companies,
currencies and macro events.  Nodes represent companies, currencies and
macroeconomic events.  Edges describe supply chain relationships between
companies, shared sectors, and country/currency exposures.  The resulting
``networkx`` graph is persisted under ``data/graphs`` by default.

The implementation intentionally relies only on widely available public
datasets represented as pandas ``DataFrame`` objects so tests can supply
small in-memory fixtures.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import networkx as nx

try:  # pragma: no cover - pandas is optional
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = object  # type: ignore


def build_knowledge_graph(
    companies: pd.DataFrame,
    supply_chain: pd.DataFrame,
    sectors: pd.DataFrame,
    countries: pd.DataFrame,
    events: List[Dict],
    path: Optional[Path] = Path("data/graphs/knowledge_graph.pkl"),
) -> nx.MultiDiGraph:
    """Build and optionally persist a heterogeneous knowledge graph.

    Parameters
    ----------
    companies:
        DataFrame containing at least a ``company`` column.
    supply_chain:
        DataFrame with ``supplier`` and ``customer`` columns describing supply
        chain links between companies.
    sectors:
        DataFrame with ``company`` and ``sector`` columns.
    countries:
        DataFrame with ``company`` and ``currency`` columns mapping companies to
        their reporting currency.
    events:
        List of dictionaries describing macroeconomic events.  Each dictionary
        should contain ``id`` and ``currency`` keys.
    path:
        Optional path where the resulting graph will be pickled.  If ``None`` the
        graph is not persisted.
    """

    g = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # Nodes -------------------------------------------------------------
    for comp in companies["company"].unique():
        g.add_node(str(comp), type="company")

    currencies: Iterable[str] = countries["currency"].unique()
    for cur in currencies:
        g.add_node(str(cur), type="currency")

    for ev in events:
        ev_id = str(ev.get("id"))
        attrs = {k: v for k, v in ev.items() if k != "id"}
        attrs.setdefault("type", "event")
        g.add_node(ev_id, **attrs)

        cur = ev.get("currency")
        if cur:
            g.add_node(str(cur), type="currency")
            g.add_edge(str(cur), ev_id, relation="country")

    # ------------------------------------------------------------------
    # Edges -------------------------------------------------------------
    # Supply chain edges - directed supplier -> customer
    for _, row in supply_chain.iterrows():
        g.add_edge(str(row["supplier"]), str(row["customer"]), relation="supply_chain")

    # Sector edges - connect companies operating in the same sector
    for sector, grp in sectors.groupby("sector"):
        comps = [str(c) for c in grp["company"].tolist()]
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                g.add_edge(comps[i], comps[j], relation="sector", sector=str(sector))
                g.add_edge(comps[j], comps[i], relation="sector", sector=str(sector))

    # Country edges - link companies to their reporting currency
    for _, row in countries.iterrows():
        g.add_edge(str(row["company"]), str(row["currency"]), relation="country")

    # Persist -----------------------------------------------------------
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(g, f)

    return g


def load_knowledge_graph(path: Path = Path("data/graphs/knowledge_graph.pkl")) -> nx.MultiDiGraph:
    """Load a previously persisted knowledge graph."""

    with Path(path).open("rb") as f:
        return pickle.load(f)


def risk_score(g: nx.MultiDiGraph, company: str) -> float:
    """Estimate a simple risk score for ``company``.

    The score equals the number of macro events connected to the company's
    reporting currency.  More events imply higher potential macro risk.
    """

    score = 0.0
    if company not in g:
        return score
    for nbr, edges in g[company].items():
        for data in edges.values():
            if data.get("relation") != "country":
                continue
            for nbr2, edges2 in g[nbr].items():
                for data2 in edges2.values():
                    if (
                        g.nodes[nbr2].get("type") == "event"
                        and data2.get("relation") == "country"
                    ):
                        score += 1.0
    return score


def opportunity_score(g: nx.MultiDiGraph, company: str) -> float:
    """Return count of same-sector connections for ``company``."""

    score = 0.0
    if company not in g:
        return score
    for edges in g[company].values():
        for data in edges.values():
            if data.get("relation") == "sector":
                score += 1.0
    return score


__all__ = [
    "build_knowledge_graph",
    "load_knowledge_graph",
    "risk_score",
    "opportunity_score",
]
