import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from models.tft import TemporalFusionTransformer, TFTConfig, QuantileLoss
from models.build_model import initialize_tft


def _train_simple(model, data, target, loss_fn):
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(5):
        preds = model(data)
        loss = loss_fn(preds, target)
        optim.zero_grad()
        loss.backward()
        optim.step()


def test_tft_quantile_prediction():
    torch.manual_seed(0)
    batch, seq, known_size = 16, 10, 3
    config = TFTConfig(
        static_size=0, known_size=known_size, observed_size=0, hidden_size=8
    )
    model = TemporalFusionTransformer(config)
    data = torch.randn(batch, seq, known_size)
    target = torch.randn(batch)
    loss_fn = QuantileLoss([0.1, 0.5, 0.9])
    _train_simple(model, data, target, loss_fn)
    preds = model.predict_quantiles(data)
    assert preds.shape == (batch, 3)
    assert model.last_attention is not None
    vi = model.variable_importance()
    assert abs(sum(vi.values()) - 1.0) < 1e-5


def test_initialize_tft_trains_attention():
    torch.manual_seed(0)
    cfg = {
        "static_size": 2,
        "known_size": 3,
        "observed_size": 1,
        "hidden_size": 8,
        "num_heads": 2,
        "quantiles": [0.1, 0.5, 0.9],
    }
    model = initialize_tft(cfg)
    known = torch.randn(4, 6, cfg["known_size"])
    static = torch.randn(4, cfg["static_size"])
    observed = torch.randn(4, 6, cfg["observed_size"])
    target = torch.randn(4)
    loss_fn = QuantileLoss(cfg["quantiles"])
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(3):
        preds = model(known, static=static, observed=observed)
        loss = loss_fn(preds, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert model.last_attention is not None
