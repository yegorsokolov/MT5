import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
from analysis.grad_monitor import GradientMonitor

def test_grad_monitor_detects_explode(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    monitor = GradientMonitor(explode=0.01, vanish=1e-9, window=1, out_dir=tmp_path)
    x = torch.tensor([[1.0]])
    y = torch.tensor([[10.0]])
    loss_fn = torch.nn.MSELoss()
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    trend, norm = monitor.track(model.parameters())
    assert trend == "explode"
    optimizer.step()
    path = monitor.plot("test")
    assert path is not None and path.exists()
