import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from models.cross_modal_transformer import CrossModalTransformer


def _f1_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)


def _train_baseline(model, x, y, epochs: int = 100) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()
    y = y.view(-1)
    for _ in range(epochs):
        opt.zero_grad()
        out = model(x).squeeze(-1)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = (model(x) > 0.5).float()
    return _f1_score(y, preds)


def _train_cross(model, price, news, y, epochs: int = 100) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()
    y = y.view(-1)
    for _ in range(epochs):
        opt.zero_grad()
        out = model(price, news)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = (model(price, news) > 0.5).float()
    return _f1_score(y, preds)


def test_cross_modal_beats_single_modal_baselines():
    torch.manual_seed(0)
    n = 200
    price = torch.randn(n, 5, 1)
    news = torch.randn(n, 3, 1)
    labels = ((price.mean(1) > 0) & (news.mean(1) > 0)).float()

    price_feat = price.view(n, -1)
    news_feat = news.view(n, -1)
    price_model = torch.nn.Sequential(torch.nn.Linear(price_feat.size(1), 1), torch.nn.Sigmoid())
    news_model = torch.nn.Sequential(torch.nn.Linear(news_feat.size(1), 1), torch.nn.Sigmoid())

    f1_price = _train_baseline(price_model, price_feat, labels)
    f1_news = _train_baseline(news_model, news_feat, labels)

    cross_model = CrossModalTransformer(
        price_dim=1, news_dim=1, d_model=16, nhead=2, num_layers=1
    )
    f1_cross = _train_cross(cross_model, price, news, labels)

    assert f1_cross > max(f1_price, f1_news)
    assert cross_model.last_attention is not None
    assert set(cross_model.last_attention.keys()) == {"price_to_news", "news_to_price"}
    price_attn = cross_model.last_attention["price_to_news"]
    news_attn = cross_model.last_attention["news_to_price"]
    assert price_attn.shape[-2:] == (price.size(1), news.size(1))
    assert news_attn.shape[-2:] == (news.size(1), price.size(1))


def test_attention_maps_are_normalised():
    torch.manual_seed(0)
    price = torch.randn(8, 4, 2)
    news = torch.randn(8, 2, 3)
    model = CrossModalTransformer(
        price_dim=2, news_dim=3, d_model=12, nhead=3, num_layers=2, dropout=0.0
    )
    model.eval()
    with torch.no_grad():
        output = model(price, news)
    assert output.shape == (price.size(0),)
    assert model.last_attention is not None
    attn = model.last_attention
    price_attn = attn["price_to_news"]
    news_attn = attn["news_to_price"]
    assert price_attn.shape == (
        len(model.layers),
        price.size(0),
        price.size(1),
        news.size(1),
    )
    assert news_attn.shape == (
        len(model.layers),
        price.size(0),
        news.size(1),
        price.size(1),
    )
    price_sums = price_attn.sum(-1)
    news_sums = news_attn.sum(-1)
    assert torch.allclose(price_sums, torch.ones_like(price_sums), atol=1e-5)
    assert torch.allclose(news_sums, torch.ones_like(news_sums), atol=1e-5)
