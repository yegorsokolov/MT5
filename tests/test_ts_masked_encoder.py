import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch import nn

# Dynamically import model_store and ts_masked_encoder
repo_root = Path(__file__).resolve().parents[1]
module_path = repo_root / "models" / "ts_masked_encoder.py"
store_path = repo_root / "models" / "model_store.py"

model_store_spec = importlib.util.spec_from_file_location("models.model_store", store_path)
model_store = importlib.util.module_from_spec(model_store_spec)
model_store_spec.loader.exec_module(model_store)

models_pkg = types.ModuleType("models")
models_pkg.model_store = model_store
sys.modules["models"] = models_pkg

spec = importlib.util.spec_from_file_location("models.ts_masked_encoder", module_path)
ts_masked = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ts_masked
spec.loader.exec_module(ts_masked)

train_ts_masked_encoder = ts_masked.train_ts_masked_encoder
initialize_model_with_ts_masked_encoder = ts_masked.initialize_model_with_ts_masked_encoder
TSMaskedEncoder = ts_masked.TSMaskedEncoder


def _make_regime_shift(n_pre: int = 200, n_shift: int = 40, seq_len: int = 8):
    t = torch.linspace(0, 20, steps=n_pre + n_shift + seq_len + 1)
    f1 = torch.sin(t)
    f2 = torch.cos(t)
    f1[n_pre:] *= 2
    features = torch.stack([f1, f2], dim=1)
    windows = torch.stack([features[i : i + seq_len] for i in range(n_pre + n_shift)])
    targets = f1[seq_len : n_pre + n_shift + seq_len].unsqueeze(-1)
    return windows[:n_pre], windows[n_pre:], targets[n_pre:]


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


def test_masked_pretraining_accelerates_adaptation(tmp_path):
    torch.manual_seed(0)
    pre_windows, shift_windows, shift_targets = _make_regime_shift()

    result = train_ts_masked_encoder(pre_windows, epochs=5, batch_size=16, store_dir=tmp_path)
    assert result.version_id is not None

    class Forecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = TSMaskedEncoder(2)
            self.head = nn.Linear(64, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h, _ = self.encoder(x)
            return self.head(h[:, -1])

    train_win = shift_windows[:10]
    train_tar = shift_targets[:10]
    val_win = shift_windows[10:]
    val_tar = shift_targets[10:]

    model_pre = Forecaster()
    initialize_model_with_ts_masked_encoder(model_pre.encoder, store_dir=tmp_path)
    model_rand = Forecaster()

    _train_one_epoch(model_pre, train_win, train_tar)
    _train_one_epoch(model_rand, train_win, train_tar)

    mse_pre = _evaluate(model_pre, val_win, val_tar)
    mse_rand = _evaluate(model_rand, val_win, val_tar)
    assert mse_pre < mse_rand
