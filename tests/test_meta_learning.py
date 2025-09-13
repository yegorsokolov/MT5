import numpy as np
import torch
from torch.utils.data import TensorDataset
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.meta_learner import (
    _LinearModel,
    fine_tune_model,
    meta_train_maml,
    _steps_to,
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


def test_meta_learning_faster_convergence():
    np.random.seed(0)
    torch.manual_seed(0)
    w1 = np.array([1.0, 1.0, -1.0])
    w2 = np.array([-1.0, 1.0, 1.0])
    tasks = [_generate_task(w1), _generate_task(w2)]
    build = lambda: _LinearModel(3)
    state = meta_train_maml(tasks, build, epochs=40)

    eval_ds, _ = _generate_task(np.array([1.0, -1.0, 1.0]))

    _, base_hist = fine_tune_model(build().state_dict(), eval_ds, build, steps=5, lr=0.1)
    _, meta_hist = fine_tune_model(state, eval_ds, build, steps=5, lr=0.1)

    assert meta_hist[-1] >= base_hist[-1]

