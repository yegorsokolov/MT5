import importlib.util
from pathlib import Path

from datetime import datetime
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "event_graph", Path(__file__).resolve().parents[1] / "analysis" / "event_graph.py"
)
event_graph_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(event_graph_mod)  # type: ignore
build_event_graph = event_graph_mod.build_event_graph


def test_event_graph_nodes_edges(tmp_path):
    events = [
        {
            "id": "e1",
            "currency": "USD",
            "event": "CPI",
            "importance": "high",
            "timestamp": datetime(2023, 1, 1),
        },
        {
            "id": "e2",
            "currency": "EUR",
            "event": "GDP",
            "importance": "low",
            "timestamp": datetime(2023, 1, 2),
        },
    ]
    macro_df = SimpleNamespace(columns=["Date", "USD_GDP", "EUR_CPI"])
    path = tmp_path / "event_graph.pkl"
    g = build_event_graph(events, macro_df, path=path)
    assert path.exists()
    # nodes
    types = {data["type"] for _, data in g.nodes(data=True)}
    assert {"event", "instrument"}.issubset(types)
    # edges
    assert g.has_edge("e1", "USD_GDP")
    assert g.has_edge("e2", "EUR_CPI")
    # weight from importance mapping
    assert g.edges["e1", "USD_GDP"]["weight"] > g.edges["e2", "EUR_CPI"]["weight"]
