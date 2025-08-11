import importlib.util
from pathlib import Path

import pandas as pd
import torch

# Load modules without importing heavy package initializers
spec = importlib.util.spec_from_file_location(
    "graph_builder", Path(__file__).resolve().parents[1] / "data" / "graph_builder.py"
)
graph_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_builder)  # type: ignore
build_correlation_graph = graph_builder.build_correlation_graph

spec_gnn = importlib.util.spec_from_file_location(
    "graph_net", Path(__file__).resolve().parents[1] / "models" / "graph_net.py"
)
graph_net_mod = importlib.util.module_from_spec(spec_gnn)
spec_gnn.loader.exec_module(graph_net_mod)  # type: ignore
GraphNet = graph_net_mod.GraphNet


def test_graph_builder_creates_edges(tmp_path):
    returns = pd.DataFrame(
        {
            "A": [0.1, 0.2, 0.3, 0.4],
            "B": [0.1, 0.2, 0.3, 0.4],
            "C": [-0.1, -0.2, -0.3, -0.4],
        }
    )
    out_path = tmp_path / "graph.pkl"
    g = build_correlation_graph(returns, threshold=0.9, path=out_path)
    assert out_path.exists()
    assert ("A", "B") in g.edges() or ("B", "A") in g.edges()


def test_graph_net_forward_pass():
    x = torch.tensor([[1.0], [2.0], [3.0]])
    edge_index = torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]])
    model = GraphNet(1, hidden_channels=4, out_channels=1)
    out = model(x, edge_index)
    assert out.shape == (3, 1)


def test_graph_net_beats_baseline():
    torch.manual_seed(0)
    n = 50
    x0 = torch.randn(n, 1)
    x1 = torch.randn(n, 1)
    x = torch.stack([x0, x1], dim=1)  # (n,2,1)
    y = torch.stack([x0, x0], dim=1)  # node1 label depends on node0 feature
    edge_index = torch.tensor([[0, 1], [1, 0]])

    # baseline linear model on individual node features
    baseline = torch.nn.Linear(1, 1)
    opt_b = torch.optim.Adam(baseline.parameters(), lr=0.1)
    for _ in range(200):
        pred = baseline(x.reshape(-1, 1)).reshape(n, 2, 1)
        loss = torch.nn.functional.mse_loss(pred, y)
        opt_b.zero_grad()
        loss.backward()
        opt_b.step()
    baseline_mse = torch.nn.functional.mse_loss(
        baseline(x.reshape(-1, 1)).reshape(n, 2, 1), y
    )

    # GraphNet with message passing
    gnn = GraphNet(1, hidden_channels=8, out_channels=1)
    opt_g = torch.optim.Adam(gnn.parameters(), lr=0.1)
    for _ in range(200):
        loss = 0.0
        for i in range(n):
            pred = gnn(x[i], edge_index)
            loss = loss + torch.nn.functional.mse_loss(pred, y[i])
        opt_g.zero_grad()
        loss.backward()
        opt_g.step()
    mse_gnn = 0.0
    for i in range(n):
        mse_gnn += torch.nn.functional.mse_loss(gnn(x[i], edge_index), y[i])
    mse_gnn /= n

    assert mse_gnn < baseline_mse
