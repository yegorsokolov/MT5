import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch import nn

# Setup model_store and contrastive_encoder modules dynamically
repo_root = Path(__file__).resolve().parents[1]
store_spec = importlib.util.spec_from_file_location(
    "models.model_store", repo_root / "models" / "model_store.py"
)
model_store = importlib.util.module_from_spec(store_spec)
store_spec.loader.exec_module(model_store)

models_pkg = types.ModuleType("models")
models_pkg.model_store = model_store
sys.modules["models"] = models_pkg

enc_spec = importlib.util.spec_from_file_location(
    "models.contrastive_encoder", repo_root / "models" / "contrastive_encoder.py"
)
contrastive = importlib.util.module_from_spec(enc_spec)
sys.modules[enc_spec.name] = contrastive
enc_spec.loader.exec_module(contrastive)
train_contrastive_encoder = contrastive.train_contrastive_encoder
initialize_model_with_contrastive = contrastive.initialize_model_with_contrastive


def _make_regime_shift(n_pre: int = 200, n_shift: int = 40, seq_len: int = 8):
    t = torch.linspace(0, 20, steps=n_pre + n_shift + seq_len + 1)
    f1 = torch.sin(t)
    f2 = torch.cos(t)
    f1[n_pre:] *= 2
    features = torch.stack([f1, f2], dim=1)
    windows = torch.stack(
        [features[i : i + seq_len].reshape(-1) for i in range(n_pre + n_shift)]
    )
    targets = f1[seq_len : n_pre + n_shift + seq_len].unsqueeze(-1)
    return windows[:n_pre], windows[n_pre:], targets[n_pre:]


class DownstreamModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.projector = nn.Sequential(nn.ReLU(), nn.Linear(16, 16))
        self.head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.head(h)


def _train_one_epoch(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    opt.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    opt.step()
    return float(loss.detach())


def test_contrastive_pretrain_improves_loss(tmp_path):
    torch.manual_seed(0)
    pre_win, shift_win, shift_tar = _make_regime_shift()
    train_contrastive_encoder(pre_win, epochs=5, batch_size=32, store_dir=tmp_path)

    model_pre = DownstreamModel(pre_win.size(1))
    model_pre = initialize_model_with_contrastive(model_pre, store_dir=tmp_path)
    model_rand = DownstreamModel(pre_win.size(1))

    loss_pre = _train_one_epoch(model_pre, shift_win[:10], shift_tar[:10])
    loss_rand = _train_one_epoch(model_rand, shift_win[:10], shift_tar[:10])
    assert loss_pre < loss_rand
