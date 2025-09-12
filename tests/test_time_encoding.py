import torch
from torch import nn

from models.multi_head import MultiHeadTransformer
from models.cross_asset_transformer import CrossAssetTransformer


def _make_dataset(n: int = 64, seq_len: int = 5):
    torch.manual_seed(0)
    times = torch.cumsum(torch.rand(n, seq_len) * 2.0, dim=1)
    delta_last = times[:, -1] - times[:, -2]
    y = (delta_last > 1.0).float()
    x = torch.zeros(n, seq_len, 1)
    x_ca = x.view(n, 1, seq_len, 1)
    times_ca = times.view(n, 1, seq_len)
    train, val = slice(0, n // 2), slice(n // 2, None)
    data = {
        "x_tr": x[train],
        "t_tr": times[train],
        "x_va": x[val],
        "t_va": times[val],
        "x_ca_tr": x_ca[train],
        "t_ca_tr": times_ca[train],
        "x_ca_va": x_ca[val],
        "t_ca_va": times_ca[val],
        "y_tr": y[train],
        "y_va": y[val],
    }
    return data


def _train_multi_head(use_time: bool) -> float:
    d = _make_dataset()
    model = MultiHeadTransformer(1, num_symbols=1, time_encoding=use_time)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    for _ in range(100):
        opt.zero_grad()
        out = model(d["x_tr"], symbol=0, times=d["t_tr"] if use_time else None)
        loss = loss_fn(out, d["y_tr"])
        loss.backward()
        opt.step()
    with torch.no_grad():
        val_out = model(d["x_va"], symbol=0, times=d["t_va"] if use_time else None)
        val_loss = loss_fn(val_out, d["y_va"])
    return float(val_loss)


def _train_cross_asset(use_time: bool) -> float:
    d = _make_dataset()
    model = CrossAssetTransformer(1, n_symbols=1, time_encoding=use_time)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(100):
        opt.zero_grad()
        out = model(d["x_ca_tr"], times=d["t_ca_tr"] if use_time else None).squeeze(-1)
        loss = loss_fn(out, d["y_tr"].unsqueeze(1))
        loss.backward()
        opt.step()
    with torch.no_grad():
        val_out = model(d["x_ca_va"], times=d["t_ca_va"] if use_time else None).squeeze(-1)
        val_loss = loss_fn(val_out, d["y_va"].unsqueeze(1))
    return float(val_loss)


def test_time_encoding_improves_loss():
    base_loss = _train_multi_head(False)
    enc_loss = _train_multi_head(True)
    assert enc_loss < base_loss

    base_loss_ca = _train_cross_asset(False)
    enc_loss_ca = _train_cross_asset(True)
    assert enc_loss_ca < base_loss_ca
