import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.meta_learner import (
    _LinearModel,
    fine_tune_model,
    meta_train_maml,
    steps_to,
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


def test_meta_init_converges_faster():
    np.random.seed(0)
    torch.manual_seed(0)

    w1 = np.array([1.0, 1.0])
    w2 = np.array([-1.0, -1.0])
    tasks = [_generate_task(w1), _generate_task(w2)]
    build = lambda: _LinearModel(2)
    meta_state = meta_train_maml(tasks, build, epochs=25)

    new_task, _ = _generate_task(np.array([1.0, -1.0]))
    _, meta_hist = fine_tune_model(meta_state, new_task, build, steps=5, lr=0.5)
    _, base_hist = fine_tune_model(build().state_dict(), new_task, build, steps=5, lr=0.5)

    assert steps_to(meta_hist) < steps_to(base_hist)
