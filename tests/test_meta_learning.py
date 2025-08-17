import numpy as np
import torch
from torch.utils.data import TensorDataset
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.meta_learning import (
    _LinearModel,
    meta_train_model,
    fine_tune_model,
)


def _generate_task(weight, n_samples: int = 40):
    X = np.random.randn(n_samples, len(weight))
    y = (X @ weight > 0).astype(float)
    split = n_samples // 2
    train_ds = TensorDataset(
        torch.tensor(X[:split], dtype=torch.float32),
        torch.tensor(y[:split], dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X[split:], dtype=torch.float32),
        torch.tensor(y[split:], dtype=torch.float32),
    )
    return train_ds, val_ds


def _steps_to(history, thr=0.9):
    for i, a in enumerate(history, 1):
        if a >= thr:
            return i
    return len(history) + 1


def test_meta_learning_faster_convergence():
    np.random.seed(0)
    torch.manual_seed(0)
    w1 = np.array([2.0, -2.0])
    w2 = np.array([-2.0, 2.0])
    tasks = [_generate_task(w1), _generate_task(w2)]
    build = lambda: _LinearModel(2)
    state = meta_train_model(tasks, build, epochs=25)

    X_new = np.random.randn(20, 2)
    y_new = (X_new @ w1 > 0).astype(float)
    dataset = TensorDataset(
        torch.tensor(X_new, dtype=torch.float32),
        torch.tensor(y_new, dtype=torch.float32),
    )

    _, base_hist = fine_tune_model(build().state_dict(), dataset, build, steps=5, lr=0.5)
    _, meta_hist = fine_tune_model(state, dataset, build, steps=5, lr=0.5)

    assert meta_hist[-1] >= base_hist[-1]
    assert _steps_to(meta_hist) <= _steps_to(base_hist)
