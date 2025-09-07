import pathlib
import sys

import torch
from torch import nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from models.cross_asset_transformer import CrossAssetTransformer


def test_forward_shape() -> None:
    model = CrossAssetTransformer(
        input_dim=4,
        n_symbols=3,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.2,
        layer_norm=True,
    )
    x = torch.randn(2, 3, 5, 4)
    out = model(x)
    assert out.shape == (2, 3, 1)


class PerSymbolLinear(nn.Module):
    def __init__(self, seq_len: int, input_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(seq_len * input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, t, f = x.shape
        return self.fc(x.view(b * s, t * f)).view(b, s, 1)


def test_cross_asset_improves_validation() -> None:
    torch.manual_seed(0)
    N, symbols, seq_len, feat = 64, 2, 3, 1
    x = torch.randn(N, symbols, seq_len, feat)
    y = torch.stack(
        [x[:, 0, -1, 0] + x[:, 1, -1, 0], x[:, 1, -1, 0] - x[:, 0, -1, 0]], dim=1
    ).unsqueeze(-1)
    train_x, train_y = x[:48], y[:48]
    val_x, val_y = x[48:], y[48:]

    model = CrossAssetTransformer(
        feat,
        symbols,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        layer_norm=True,
    )
    baseline = PerSymbolLinear(seq_len, feat)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    opt_b = torch.optim.Adam(baseline.parameters(), lr=0.01)
    for _ in range(50):
        opt.zero_grad()
        loss = ((model(train_x) - train_y) ** 2).mean()
        loss.backward()
        opt.step()

        opt_b.zero_grad()
        loss_b = ((baseline(train_x) - train_y) ** 2).mean()
        loss_b.backward()
        opt_b.step()

    val_loss = ((model(val_x) - val_y) ** 2).mean().item()
    val_loss_base = ((baseline(val_x) - val_y) ** 2).mean().item()
    assert val_loss < val_loss_base
