import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")

from models.cross_modal_transformer import CrossModalTransformer
from mt5.train_utils import prepare_modal_arrays


def _train_linear(x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor, epochs: int = 200) -> float:
    model = torch.nn.Sequential(torch.nn.Linear(x_train.shape[1], 1), torch.nn.Sigmoid())
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()
    for _ in range(epochs):
        opt.zero_grad()
        preds = model(x_train).squeeze()
        loss = loss_fn(preds, y_train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds_val = (model(x_val).squeeze() > 0.5).float()
    true = y_val.detach().cpu()
    pred = preds_val.detach().cpu()
    tp = ((true == 1) & (pred == 1)).sum().item()
    fp = ((true == 0) & (pred == 1)).sum().item()
    fn = ((true == 1) & (pred == 0)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)


def _train_cross_modal(
    price_train: torch.Tensor,
    news_train: torch.Tensor,
    y_train: torch.Tensor,
    price_val: torch.Tensor,
    news_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 200,
) -> float:
    model = CrossModalTransformer(
        price_dim=price_train.shape[-1],
        news_dim=news_train.shape[-1],
        d_model=32,
        nhead=2,
        num_layers=2,
        dropout=0.1,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()
    for _ in range(epochs):
        opt.zero_grad()
        preds = model(price_train, news_train)
        loss = loss_fn(preds, y_train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds_val = (model(price_val, news_val) > 0.5).float()
    true = y_val.detach().cpu()
    pred = preds_val.detach().cpu()
    tp = ((true == 1) & (pred == 1)).sum().item()
    fp = ((true == 0) & (pred == 1)).sum().item()
    fn = ((true == 1) & (pred == 0)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)


def test_cross_modal_beats_single_modal_baselines():
    torch.manual_seed(0)
    np.random.seed(0)
    n_samples = 240
    window = 5
    news_dim = 4

    price_series = np.random.randn(n_samples + window)
    data = {}
    for step in range(window):
        data[f"price_window_{step}"] = price_series[step : step + n_samples]
    news_values = np.random.randn(n_samples, news_dim)
    for i in range(news_dim):
        data[f"news_emb_{i}"] = news_values[:, i]

    price_signal = sum(data[f"price_window_{step}"] for step in range(window)) / window
    news_signal = news_values.mean(axis=1)
    labels = ((price_signal > 0) & (news_signal > 0)).astype(float)
    df = pd.DataFrame(data)
    df["tb_label"] = labels

    arrays = prepare_modal_arrays(df, df["tb_label"].to_numpy())
    assert arrays is not None
    price_tensor, news_tensor, label_tensor, _, _, _ = arrays
    assert label_tensor is not None

    # keep chronological ordering for train/validation split
    split = int(len(price_tensor) * 0.75)
    price_train_np, price_val_np = price_tensor[:split], price_tensor[split:]
    news_train_np, news_val_np = news_tensor[:split], news_tensor[split:]
    y_train_np, y_val_np = label_tensor[:split], label_tensor[split:]

    price_train = torch.tensor(price_train_np.reshape(len(price_train_np), -1), dtype=torch.float32)
    price_val = torch.tensor(price_val_np.reshape(len(price_val_np), -1), dtype=torch.float32)
    news_train = torch.tensor(news_train_np.reshape(len(news_train_np), -1), dtype=torch.float32)
    news_val = torch.tensor(news_val_np.reshape(len(news_val_np), -1), dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.float32)

    f1_price = _train_linear(price_train, y_train, price_val, y_val)
    f1_news = _train_linear(news_train, y_train, news_val, y_val)

    price_train_modal = torch.tensor(price_train_np, dtype=torch.float32)
    price_val_modal = torch.tensor(price_val_np, dtype=torch.float32)
    news_train_modal = torch.tensor(news_train_np, dtype=torch.float32)
    news_val_modal = torch.tensor(news_val_np, dtype=torch.float32)
    f1_cross = _train_cross_modal(
        price_train_modal,
        news_train_modal,
        y_train,
        price_val_modal,
        news_val_modal,
        y_val,
    )

    assert f1_cross > max(f1_price, f1_news)
