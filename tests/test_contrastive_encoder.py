import math
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import importlib.util
import types
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
module_path = repo_root / "models" / "contrastive_encoder.py"
store_path = repo_root / "models" / "model_store.py"

model_store_spec = importlib.util.spec_from_file_location("models.model_store", store_path)
model_store = importlib.util.module_from_spec(model_store_spec)
model_store_spec.loader.exec_module(model_store)

models_pkg = types.ModuleType("models")
models_pkg.model_store = model_store
sys.modules["models"] = models_pkg

spec = importlib.util.spec_from_file_location("models.contrastive_encoder", module_path)
contrastive_encoder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = contrastive_encoder
spec.loader.exec_module(contrastive_encoder)
train_contrastive_encoder = contrastive_encoder.train_contrastive_encoder
load_pretrained_contrastive_encoder = contrastive_encoder.load_pretrained_contrastive_encoder


def _make_circle_data(radii, n_per_class, seed=0):
    g = torch.Generator().manual_seed(seed)
    data = []
    labels = []
    for idx, r in enumerate(radii):
        angles = torch.rand(n_per_class, generator=g) * 2 * math.pi
        x = torch.stack([r * torch.cos(angles), r * torch.sin(angles)], dim=1)
        data.append(x)
        labels.append(torch.full((n_per_class,), idx, dtype=torch.long))
    return torch.cat(data), torch.cat(labels)


def _train_linear(x, y, epochs=100):
    model = nn.Linear(x.size(1), int(y.max().item()) + 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for bx, by in loader:
            pred = model(bx)
            loss = loss_fn(pred, by)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        preds = model(x).argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return acc


def test_contrastive_encoder_improves_accuracy(tmp_path):
    radii = [1.0, 2.0, 3.0]
    X, y = _make_circle_data(radii, n_per_class=40)

    base_acc = _train_linear(X, y)

    train_contrastive_encoder(X, epochs=40, batch_size=32, store_dir=tmp_path)
    state = load_pretrained_contrastive_encoder(store_dir=tmp_path)
    assert state is not None
    contrastive = contrastive_encoder.ContrastiveEncoder(2)
    contrastive.load_state_dict(state)
    with torch.no_grad():
        Z = contrastive.encoder(X)
    acc = _train_linear(Z, y)
    assert acc > base_acc + 0.1
