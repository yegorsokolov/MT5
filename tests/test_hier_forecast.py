import pytest
import sys
import pathlib

torch = pytest.importorskip("torch")
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "models"))
from hier_forecast import HierarchicalForecaster

def test_hier_forecast_consistency():
    torch.manual_seed(0)
    horizons = [1, 3]
    model = HierarchicalForecaster(input_size=1, horizons=horizons, d_model=8)
    X = torch.randn(128, 10, 1)
    y_base = (X[:, -1, 0] > 0).float()
    Y = torch.stack([y_base for _ in horizons], dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = torch.nn.BCELoss()
    for _ in range(100):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        preds = model(X)
    assert preds.shape == (128, len(horizons))
    diff = torch.abs(preds[:, 0] - preds[:, -1]).mean().item()
    assert diff < 0.1
