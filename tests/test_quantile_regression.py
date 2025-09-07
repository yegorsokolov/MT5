import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from models.quantile_regression import NeuralQuantile


def test_dropout_behavior():
    torch.manual_seed(0)
    nq = NeuralQuantile(input_dim=2, alphas=[0.5], dropout=0.5)
    net = nq._build_net()
    x = torch.ones(10, 2)
    net.train()
    out1 = net(x)
    out2 = net(x)
    assert not torch.allclose(out1, out2)
    net.eval()
    out3 = net(x)
    out4 = net(x)
    assert torch.allclose(out3, out4)


def test_early_stopping_triggers():
    torch.manual_seed(0)
    X = np.zeros((100, 2), dtype=np.float32)
    y = np.zeros(100, dtype=np.float32)
    nq = NeuralQuantile(input_dim=2, alphas=[0.5], epochs=20, patience=1)
    nq.fit(X, y, val=(X, y))
    assert nq.epochs_trained[0.5] < nq.epochs
