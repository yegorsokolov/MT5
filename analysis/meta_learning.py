"""Model-agnostic meta-learning utilities.

This module implements a lightweight version of the MAML algorithm for
PyTorch models and provides convenience wrappers for the core models used
in the project.  The implementation is intentionally simple so it can run
within the unit tests where only small synthetic datasets are required.

The key entry points are :func:`meta_train_model` which performs the meta
update over a list of tasks and :func:`fine_tune_model` which adapts the
meta-initialised weights to a new task with only a few gradient steps.

Although the real project contains complex models such as LightGBM
boosters, transformers and reinforcement learning policies, in the
context of the tests we operate on small neural networks which share the
same interface.  The wrappers ``meta_train_lgbm``, ``meta_train_transformer``
and ``meta_train_policy`` simply call :func:`meta_train_model` with the
appropriate model builders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:  # optional but typically available in the project
    from models import model_store  # type: ignore
except Exception:  # pragma: no cover - model_store may not be available in tests
    model_store = None  # type: ignore


# ---------------------------------------------------------------------------
# Helper models and datasets
# ---------------------------------------------------------------------------


class _LinearModel(nn.Module):
    """Simple logistic regression style model used in tests.

    The model mirrors the interface of more complex models in the project
    and is sufficient to demonstrate the benefits of meta-learning.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return self.fc(x).squeeze(1)


# ---------------------------------------------------------------------------
# Core MAML routines
# ---------------------------------------------------------------------------


def meta_train_model(
    tasks: Sequence[Tuple[TensorDataset, TensorDataset]],
    build_model: Callable[[], nn.Module],
    inner_lr: float = 0.1,
    meta_lr: float = 0.01,
    inner_steps: int = 1,
    epochs: int = 5,
) -> dict:
    """Meta-train ``build_model`` using the MAML algorithm.

    Parameters
    ----------
    tasks:
        Sequence of ``(train_ds, val_ds)`` pairs representing different
        regimes/tasks.
    build_model:
        Callable that returns a new uninitialised model instance.
    inner_lr, meta_lr:
        Learning rates for the inner and outer loops.
    inner_steps:
        Number of gradient steps taken during task adaptation.
    epochs:
        Number of meta-epochs.

    Returns
    -------
    dict
        A state dictionary containing the meta-initialised weights.
    """

    loss_fn = nn.BCEWithLogitsLoss()
    meta_model = build_model()

    for _ in range(epochs):
        meta_grads = [torch.zeros_like(p) for p in meta_model.parameters()]
        for train_ds, val_ds in tasks:
            model = build_model()
            model.load_state_dict(meta_model.state_dict())
            opt = torch.optim.SGD(model.parameters(), lr=inner_lr)
            train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=True)
            for _ in range(inner_steps):
                xb, yb = next(iter(train_loader))
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
            x_val, y_val = val_ds.tensors
            loss_val = loss_fn(model(x_val), y_val)
            grads = torch.autograd.grad(loss_val, model.parameters())
            for g, mg in zip(grads, meta_grads):
                mg += g
        for p, g in zip(meta_model.parameters(), meta_grads):
            p.data -= meta_lr * g / len(tasks)
    return meta_model.state_dict()


def fine_tune_model(
    state_dict: dict,
    dataset: TensorDataset,
    build_model: Callable[[], nn.Module],
    steps: int = 5,
    lr: float = 0.1,
) -> Tuple[dict, List[float]]:
    """Fine-tune a model initialised with ``state_dict`` on ``dataset``.

    The function returns the updated state dict and a list of accuracy
    values after each gradient step, allowing tests to inspect the speed of
    convergence.
    """

    model = build_model()
    model.load_state_dict(state_dict)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    X_all, y_all = dataset.tensors
    history: List[float] = []
    for _ in range(steps):
        xb, yb = next(iter(loader))
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        with torch.no_grad():
            preds = (torch.sigmoid(model(X_all)) > 0.5).float()
            acc = (preds == y_all).float().mean().item()
        history.append(acc)
    return model.state_dict(), history


# ---------------------------------------------------------------------------
# Convenience wrappers for project models
# ---------------------------------------------------------------------------


def _split_by_regime(
    X: np.ndarray, y: np.ndarray, regimes: Iterable[int]
) -> List[Tuple[TensorDataset, TensorDataset]]:
    tasks: List[Tuple[TensorDataset, TensorDataset]] = []
    regimes = np.asarray(list(regimes))
    for r in np.unique(regimes):
        mask = regimes == r
        X_r, y_r = X[mask], y[mask]
        split = max(len(X_r) // 2, 1)
        train_ds = TensorDataset(
            torch.tensor(X_r[:split], dtype=torch.float32),
            torch.tensor(y_r[:split], dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_r[split:], dtype=torch.float32),
            torch.tensor(y_r[split:], dtype=torch.float32),
        )
        tasks.append((train_ds, val_ds))
    return tasks


def meta_train_lgbm(df, features: List[str]) -> dict:
    """Meta-train a linear approximation of an LGBM model.

    The real project uses LightGBM which is not differentiable.  For meta
    learning we approximate it with a small neural network over the selected
    features.  The function expects ``df`` to contain a ``market_regime``
    column which defines the tasks.
    """

    X = df[features].values.astype(np.float32)
    y = (df["return"].shift(-1) > 0).astype(np.float32).values[:-1]
    X = X[:-1]
    tasks = _split_by_regime(X, y, df["market_regime"].iloc[:-1])
    build = lambda: _LinearModel(len(features))
    return meta_train_model(tasks, build)


def meta_train_transformer(tasks: Sequence[Tuple[TensorDataset, TensorDataset]], build_model: Callable[[], nn.Module]) -> dict:
    """Thin wrapper used by :mod:`train_nn` to perform meta-training."""

    return meta_train_model(tasks, build_model)


def meta_train_policy(tasks: Sequence[Tuple[TensorDataset, TensorDataset]], build_model: Callable[[], nn.Module]) -> dict:
    """Thin wrapper used by :mod:`train_rl` to perform meta-training."""

    return meta_train_model(tasks, build_model)


def save_meta_weights(state: dict, model_name: str, regime: str = "meta") -> None:
    """Persist ``state`` to the :mod:`model_store` tagged by ``regime``."""

    if model_store is None:  # pragma: no cover - store not available
        return
    path = Path(f"{model_name}_{regime}.pt")
    torch.save(state, path)
    model_store.save_model(
        path,
        training_config={"model": model_name, "regime": regime, "meta": True},
        performance={},
    )


def load_meta_weights(model_name: str, regime: str = "meta") -> dict:
    """Load the latest meta-weights for ``model_name`` from the store."""

    if model_store is None:  # pragma: no cover - store not available
        raise FileNotFoundError("model_store not available")
    for meta in reversed(model_store.list_versions()):
        cfg = meta.get("training_config", {})
        if (
            cfg.get("model") == model_name
            and cfg.get("regime") == regime
            and cfg.get("meta")
        ):
            state, _ = model_store.load_model(meta["version_id"])
            if isinstance(state, dict):
                return state
            return torch.load(state)  # type: ignore[arg-type]
    raise FileNotFoundError("No meta weights found")


__all__ = [
    "meta_train_model",
    "fine_tune_model",
    "meta_train_lgbm",
    "meta_train_transformer",
    "meta_train_policy",
    "save_meta_weights",
    "load_meta_weights",
]
