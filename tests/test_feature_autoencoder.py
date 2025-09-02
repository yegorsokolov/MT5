import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch import nn

# Dynamically import model_store and feature_autoencoder to avoid optional deps
repo_root = Path(__file__).resolve().parents[1]
module_path = repo_root / "models" / "feature_autoencoder.py"
store_path = repo_root / "models" / "model_store.py"

model_store_spec = importlib.util.spec_from_file_location("models.model_store", store_path)
model_store = importlib.util.module_from_spec(model_store_spec)
model_store_spec.loader.exec_module(model_store)

models_pkg = types.ModuleType("models")
models_pkg.model_store = model_store
sys.modules["models"] = models_pkg

spec = importlib.util.spec_from_file_location("models.feature_autoencoder", module_path)
fae = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fae
spec.loader.exec_module(fae)

train_feature_autoencoder = fae.train_feature_autoencoder
initialize_model_with_feature_ae = fae.initialize_model_with_feature_ae
FeatureAutoencoder = fae.FeatureAutoencoder


def _make_regime_data(n_pre: int = 200, n_shift: int = 40, dim: int = 6):
    torch.manual_seed(0)
    x_pre = torch.randn(n_pre, dim)
    x_shift = torch.randn(n_shift, dim)
    y_pre = x_pre[:, 0] - x_pre[:, 1] + 0.1 * torch.randn(n_pre)
    y_shift = -x_shift[:, 0] + x_shift[:, 1] + 0.1 * torch.randn(n_shift)
    return x_pre, y_pre.unsqueeze(-1), x_shift, y_shift.unsqueeze(-1)


def _train_one_epoch(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    opt.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    opt.step()
    return float(loss.detach())


def _evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return nn.functional.mse_loss(model(x), y).item()


def test_feature_autoencoder_pretraining_accelerates_adaptation(tmp_path):
    x_pre, y_pre, x_shift, y_shift = _make_regime_data()

    result = train_feature_autoencoder(torch.cat([x_pre, x_shift]), epochs=5, store_dir=tmp_path)
    assert result.version_id is not None

    class Forecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = FeatureAutoencoder(x_pre.shape[1], embed_dim=2).encoder
            self.head = nn.Linear(2, 1)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = self.encoder(x)
            return self.head(z)

    train_x = x_shift[:10]
    train_y = y_shift[:10]
    val_x = x_shift[10:]
    val_y = y_shift[10:]

    model_pre = Forecaster()
    initialize_model_with_feature_ae(model_pre.encoder, store_dir=tmp_path)
    for p in model_pre.encoder.parameters():
        p.requires_grad = False

    model_rand = Forecaster()
    for p in model_rand.encoder.parameters():
        p.requires_grad = False

    _train_one_epoch(model_pre, train_x, train_y)
    _train_one_epoch(model_rand, train_x, train_y)

    mse_pre = _evaluate(model_pre, val_x, val_y)
    mse_rand = _evaluate(model_rand, val_x, val_y)
    assert mse_pre < mse_rand
