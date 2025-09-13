"""Example training script for meta-learning across assets.

The goal of this script is not to provide production ready training but to
showcase how the meta-learning utilities in :mod:`models.meta_learner` can be
used.  It generates a few synthetic "assets" (linear classification problems),
performs meta-training to learn a good initialisation and then fine-tunes the
model on each asset while logging the adaptation speed.

Running the script is optional for the unit tests but serves as reference
documentation for developers experimenting with meta-learning techniques.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

from models.meta_learner import (
    _LinearModel,
    fine_tune_model,
    meta_train_maml,
    _steps_to,
)

logger = logging.getLogger(__name__)


def _generate_task(weight: np.ndarray, n_samples: int = 40) -> Tuple[TensorDataset, TensorDataset]:
    """Create a binary classification task defined by ``weight``."""

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


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    np.random.seed(0)
    torch.manual_seed(0)

    # ------------------------------------------------------------------
    # Create a set of synthetic assets/tasks
    # ------------------------------------------------------------------
    weights = [np.array([2.0, -2.0]), np.array([-2.0, 2.0]), np.array([1.5, 1.5])]
    tasks = [_generate_task(w) for w in weights]

    build = lambda: _LinearModel(2)

    # ------------------------------------------------------------------
    # Meta-training
    # ------------------------------------------------------------------
    meta_state = meta_train_maml(tasks, build, epochs=50)
    logger.info("Meta-training complete")

    # ------------------------------------------------------------------
    # Fine-tune per asset and log adaptation speed
    # ------------------------------------------------------------------
    for i, (train_ds, _) in enumerate(tasks):
        _, history = fine_tune_model(meta_state, train_ds, build, steps=5, lr=0.5)
        logger.info("Asset %d accuracy after each step: %s (reached 90%% in %d steps)",
                    i, [round(h, 3) for h in history], _steps_to(history))


if __name__ == "__main__":  # pragma: no cover - manual execution script
    main()

