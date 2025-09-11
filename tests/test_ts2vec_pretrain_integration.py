import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch import nn

# Dynamically load model_store and ts2vec modules
repo_root = Path(__file__).resolve().parents[1]
model_store_spec = importlib.util.spec_from_file_location(
    "models.model_store", repo_root / "models" / "model_store.py"
)
model_store = importlib.util.module_from_spec(model_store_spec)
model_store_spec.loader.exec_module(model_store)

models_pkg = types.ModuleType("models")
models_pkg.model_store = model_store
sys.modules["models"] = models_pkg

spec = importlib.util.spec_from_file_location(
    "models.ts2vec", repo_root / "models" / "ts2vec.py"
)
ts2vec = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ts2vec
spec.loader.exec_module(ts2vec)

train_ts2vec_encoder = ts2vec.train_ts2vec_encoder
TS2VecEncoder = ts2vec.TS2VecEncoder

ts_masked_stub = types.ModuleType("models.ts_masked_encoder")
def _noop(model, store_dir=None):
    return model
ts_masked_stub.initialize_model_with_ts_masked_encoder = _noop
sys.modules["models.ts_masked_encoder"] = ts_masked_stub

contrastive_stub = types.ModuleType("models.contrastive_encoder")
contrastive_stub.initialize_model_with_contrastive = _noop
sys.modules["models.contrastive_encoder"] = contrastive_stub

# Stub dependencies for build_model
res_mod = types.ModuleType("utils.resource_monitor")
res_mod.monitor = types.SimpleNamespace(capabilities=types.SimpleNamespace(has_gpu=False, cpus=4))
sys.modules["utils.resource_monitor"] = res_mod

graph_mod = types.ModuleType("models.graph_net")
class GraphNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x
graph_mod.GraphNet = GraphNet
sys.modules["models.graph_net"] = graph_mod

cross_mod = types.ModuleType("models.cross_modal_transformer")
class CrossModalTransformer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, *args, **kwargs):
        return torch.zeros(1)
cross_mod.CrossModalTransformer = CrossModalTransformer
sys.modules["models.cross_modal_transformer"] = cross_mod

class MultiHeadTransformer(TS2VecEncoder):
    def __init__(self, input_size: int, num_symbols: int, **kwargs):
        super().__init__(input_size)
        self.head = nn.Linear(32, 1)
    def forward(self, x: torch.Tensor, symbol: int) -> torch.Tensor:
        h, _ = super().forward(x)
        return self.head(h[:, -1])

multi_head_mod = types.ModuleType("models.multi_head")
multi_head_mod.MultiHeadTransformer = MultiHeadTransformer
sys.modules["models.multi_head"] = multi_head_mod

# Import build_model after stubbing
build_spec = importlib.util.spec_from_file_location(
    "models.build_model", repo_root / "models" / "build_model.py"
)
build_module = importlib.util.module_from_spec(build_spec)
sys.modules["models.build_model"] = build_module
build_spec.loader.exec_module(build_module)
build_model = build_module.build_model


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
    pred = model(x, symbol=0)
    loss = loss_fn(pred, y.squeeze(-1))
    loss.backward()
    opt.step()
    return float(loss.detach())


def test_build_model_uses_ts2vec_pretrain(tmp_path):
    torch.manual_seed(0)
    pre_win, shift_win, shift_tar = _make_regime_shift()
    train_ts2vec_encoder(pre_win, epochs=5, batch_size=16, store_dir=tmp_path)

    model_pre = build_model(2, {"use_ts2vec_pretrain": True}, num_symbols=1)
    model_rand = build_model(2, {"use_ts2vec_pretrain": False}, num_symbols=1)

    loss_pre = _train_one_epoch(model_pre, shift_win[:10], shift_tar[:10])
    loss_rand = _train_one_epoch(model_rand, shift_win[:10], shift_tar[:10])
    assert loss_pre < loss_rand
