import sys
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.meta_learner import MetaLearner


class SimpleNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(2, 4), torch.nn.ReLU()
        )
        self.head = torch.nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.shared(x))).squeeze(1)


def generate_dataset(n: int, shift: float = 0.0) -> TensorDataset:
    X = torch.randn(n, 2)
    y = (X.sum(dim=1) + shift > 0).float()
    return TensorDataset(X, y)


def train_model(model: torch.nn.Module, dataset: TensorDataset, epochs: int = 5):
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = torch.nn.BCELoss()
    history = []
    X_all, y_all = dataset.tensors
    for _ in range(epochs):
        for X, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
        with torch.no_grad():
            preds = (model(X_all) > 0.5).float()
            acc = (preds == y_all).float().mean().item()
        history.append(acc)
    return history


def test_transfer_learning_improves_accuracy():
    torch.manual_seed(0)
    base_model = SimpleNet()
    pretrain_ds = generate_dataset(500)
    train_model(base_model, pretrain_ds, epochs=10)

    small_ds = generate_dataset(50, shift=0.1)

    meta = MetaLearner(base_model)
    transfer_hist = meta.fine_tune(small_ds, epochs=5, lr=0.05)

    scratch_model = SimpleNet()
    scratch_hist = train_model(scratch_model, small_ds, epochs=5)

    assert transfer_hist[0] > scratch_hist[0]
    assert transfer_hist[-1] > scratch_hist[-1]
