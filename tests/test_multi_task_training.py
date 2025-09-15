import numpy as np
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.multi_head import MultiHeadTransformer


def _f1_score(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2 * tp / denom


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _train_single_task(X, target, task: str):
    model = MultiHeadTransformer(2, num_symbols=1, d_model=16, nhead=2, num_layers=1, dropout=0.0, horizons=[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    x_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(target, dtype=torch.float32)
    for _ in range(50):
        opt.zero_grad()
        out = model(x_tensor, 0)
        if task == "direction":
            loss = torch.nn.functional.binary_cross_entropy(out['direction_1'], y_t)
        else:
            loss = torch.nn.functional.mse_loss(out['abs_return_1'], y_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = model(x_tensor, 0)
    return out


def test_joint_training_improves_metrics():
    np.random.seed(0)
    torch.manual_seed(0)
    n = 200
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    X = np.stack([x1, x2], axis=1)
    direction = (x1 + x2 + 0.1 * np.random.randn(n) > 0).astype(int)
    abs_return = np.abs(x1 + x2) + 0.1 * np.random.randn(n)
    vol = np.abs(x1 - x2) + 0.1 * np.random.randn(n)

    # Baseline models trained separately
    dir_out = _train_single_task(X, direction, "direction")
    f1_baseline = _f1_score(direction, (dir_out['direction_1'] > 0.5).int().numpy())
    abs_out = _train_single_task(X, abs_return, "abs")
    rmse_baseline = _rmse(abs_return, abs_out['abs_return_1'].numpy())

    # Joint training across all tasks
    model = MultiHeadTransformer(2, num_symbols=1, d_model=16, nhead=2, num_layers=1, dropout=0.0, horizons=[1])
    x_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_dir = torch.tensor(direction, dtype=torch.float32)
    y_abs = torch.tensor(abs_return, dtype=torch.float32)
    y_vol = torch.tensor(vol, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        out = model(x_tensor, 0)
        loss = (
            torch.nn.functional.binary_cross_entropy(out['direction_1'], y_dir)
            + torch.nn.functional.mse_loss(out['abs_return_1'], y_abs)
            + torch.nn.functional.mse_loss(out['volatility_1'], y_vol)
        )
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = model(x_tensor, 0)
    f1_joint = _f1_score(direction, (out['direction_1'] > 0.5).int().numpy())
    rmse_joint = _rmse(abs_return, out['abs_return_1'].numpy())

    assert f1_joint > f1_baseline
    assert rmse_joint < rmse_baseline
