import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.diff_backtest import DiffBacktestLoss
from models.quantile_regression import NeuralQuantile


def finite_difference(loss_fn, param, idx, eps=1e-4):
    w = param.data.clone()
    param.data[idx] = w[idx] + eps
    loss_pos = loss_fn().item()
    param.data[idx] = w[idx] - eps
    loss_neg = loss_fn().item()
    param.data[idx] = w[idx]
    return (loss_pos - loss_neg) / (2 * eps)


def test_strategy_controller_gradients_match_fd():
    torch.manual_seed(0)
    prices = torch.tensor([1.0, 1.01, 1.02, 1.03], dtype=torch.float32)
    feats = torch.randn(len(prices), 1)
    net = torch.nn.Linear(1, 3)
    loss_fn = DiffBacktestLoss(slippage=1e-3)

    def loss_wrapper():
        logits = net(feats)
        return loss_fn(prices, logits)

    net.zero_grad()
    loss = loss_wrapper()
    loss.backward()
    idx = (0, 0)
    grad = net.weight.grad[idx].item()
    fd = finite_difference(loss_wrapper, net.weight, idx)
    assert grad == pytest.approx(fd, rel=1e-2, abs=1e-2)


def test_neural_quantile_backprop():
    torch.manual_seed(0)
    prices = torch.tensor([1.0, 1.01, 1.02, 1.03], dtype=torch.float32)
    feats = torch.randn(len(prices), 2)
    nq = NeuralQuantile(input_dim=2, alphas=[0.5])
    net = nq._build_net()
    loss_fn = DiffBacktestLoss(slippage=1e-3)

    param = next(net.parameters())
    idx = (0, 0)

    def loss_wrapper():
        out = net(feats).squeeze(-1)
        pos = torch.tanh(out)
        return loss_fn(prices, pos)

    net.zero_grad()
    loss = loss_wrapper()
    loss.backward()
    grad = param.grad[idx].item()
    fd = finite_difference(loss_wrapper, param, idx)
    assert grad == pytest.approx(fd, rel=1e-2, abs=1e-2)
