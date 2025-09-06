import importlib.util
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Load knowledge graph module
kg_spec = importlib.util.spec_from_file_location(
    "knowledge_graph", Path(__file__).resolve().parents[1] / "analysis" / "knowledge_graph.py"
)
kg_mod = importlib.util.module_from_spec(kg_spec)
kg_spec.loader.exec_module(kg_mod)  # type: ignore
build_knowledge_graph = kg_mod.build_knowledge_graph

# Import router and then patch its load_knowledge_graph function
from strategy import router as router_mod  # type: ignore
StrategyRouter = router_mod.StrategyRouter


def test_router_augments_with_graph(tmp_path, monkeypatch):
    companies = pd.DataFrame({"company": ["A", "B"]})
    supply = pd.DataFrame({"supplier": ["A"], "customer": ["B"]})
    sectors = pd.DataFrame({"company": ["A", "B"], "sector": ["tech", "tech"]})
    countries = pd.DataFrame({"company": ["A", "B"], "currency": ["USD", "EUR"]})
    events = [{"id": "e1", "currency": "USD", "name": "FOMC"}]
    path = tmp_path / "knowledge_graph.pkl"
    g = build_knowledge_graph(companies, supply, sectors, countries, events, path=path)

    monkeypatch.setattr(router_mod, "load_knowledge_graph", lambda: g)

    router = StrategyRouter(algorithms={"a": lambda f: 0.0})
    features = {"company": "A", "volatility": 0.1, "trend_strength": 0.2, "regime": 0.0}
    router.select(features)
    assert features["graph_risk"] == 1.0
    assert features["graph_opportunity"] == 1.0
