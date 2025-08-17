import torch
from models.tft import TemporalFusionTransformer, TFTConfig, QuantileLoss


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
    config = TFTConfig(static_size=0, known_size=known_size, observed_size=0, hidden_size=8)
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
