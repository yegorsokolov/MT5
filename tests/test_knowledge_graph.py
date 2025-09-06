import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "knowledge_graph", Path(__file__).resolve().parents[1] / "analysis" / "knowledge_graph.py"
)
kg_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kg_mod)  # type: ignore
build_knowledge_graph = kg_mod.build_knowledge_graph
risk_score = kg_mod.risk_score
opportunity_score = kg_mod.opportunity_score


def test_knowledge_graph_build_and_query(tmp_path):
    companies = pd.DataFrame({"company": ["A", "B"]})
    supply = pd.DataFrame({"supplier": ["A"], "customer": ["B"]})
    sectors = pd.DataFrame({"company": ["A", "B"], "sector": ["tech", "tech"]})
    countries = pd.DataFrame({"company": ["A", "B"], "currency": ["USD", "EUR"]})
    events = [{"id": "e1", "currency": "USD", "name": "FOMC"}]
    path = tmp_path / "knowledge_graph.pkl"

    g = build_knowledge_graph(companies, supply, sectors, countries, events, path=path)
    assert path.exists()
    # nodes
    assert g.nodes["A"]["type"] == "company"
    assert g.nodes["USD"]["type"] == "currency"
    assert g.nodes["e1"]["type"] == "event"
    # edges
    assert g.has_edge("A", "B")
    edge_data = g.get_edge_data("A", "B") or {}
    assert any(d.get("relation") == "supply_chain" for d in edge_data.values())
    assert g.has_edge("A", "USD")
    edge_data = g.get_edge_data("A", "USD") or {}
    assert any(d.get("relation") == "country" for d in edge_data.values())
    assert g.has_edge("USD", "e1")

    # queries
    assert risk_score(g, "A") == 1.0
    assert opportunity_score(g, "A") == 1.0
