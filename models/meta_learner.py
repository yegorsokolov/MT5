"""Utilities for model-agnostic meta learning.

This module contains two distinct sets of functionality:

* ``MetaLearner`` – a light‑weight helper used in the project for classical
  transfer learning where a pre‑trained model is adapted to a new symbol by
  fine‑tuning the final layer.  This class existed prior to the meta‑learning
  work and is kept intact to preserve backwards compatibility with existing
  tests (see :mod:`tests.test_transfer_learning`).

* Meta‑learning routines (MAML and Reptile) that learn an initialisation across
  a collection of tasks/assets.  These functions are intentionally simple so
  they can operate on small synthetic datasets used within the unit tests.  The
  goal is to provide a good starting set of weights that adapts quicker than a
  random initialisation.

The algorithms implemented here are **first‑order** variants – second order
derivatives are avoided to keep the implementation compact and to work on
machines without advanced autograd features.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, Iterable, List, Sequence, Tuple

from . import model_store


# ---------------------------------------------------------------------------
# Transfer learning helper (existing functionality)
# ---------------------------------------------------------------------------


class MetaLearner:
    """Fine-tune a pre-trained model for a new symbol.

    The learner freezes all layers except the last child module and
    trains remaining parameters on a small dataset.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    @staticmethod
    def _load_latest_state_dict(symbol: str) -> dict | None:
        """Return the latest state_dict for a symbol from the model store."""
        try:
            versions = model_store.list_versions()
        except Exception:  # pragma: no cover - store may not exist
            return None
        for meta in reversed(versions):
            cfg = meta.get("training_config", {})
            syms = cfg.get("symbols") or [cfg.get("symbol")]
            if syms and symbol in syms:
                state, _ = model_store.load_model(meta["version_id"])
                if isinstance(state, dict):
                    return state
        return None

    @classmethod
    def from_symbol(
        cls, symbol: str, builder: Callable[[], torch.nn.Module]
    ) -> "MetaLearner":
        """Construct a MetaLearner initialised from a donor symbol."""
        model = builder()
        state_dict = cls._load_latest_state_dict(symbol)
        if state_dict:
            model.load_state_dict(state_dict, strict=False)
        return cls(model)

    def freeze_shared_layers(self) -> None:
        children = list(self.model.children())
        for child in children[:-1]:
            for p in child.parameters():
                p.requires_grad = False

    def fine_tune(
        self,
        dataset: TensorDataset,
        epochs: int = 5,
        lr: float = 1e-3,
        batch_size: int = 32,
        device: str | None = None,
    ) -> list[float]:
        """Fine-tune the model and return accuracy per epoch."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.freeze_shared_layers()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimiser = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr
        )
        loss_fn = torch.nn.BCELoss()
        X_all, y_all = dataset.tensors
        history: list[float] = []
        for _ in range(epochs):
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                optimiser.zero_grad()
                out = self.model(X)
                loss = loss_fn(out, y)
                loss.backward()
                optimiser.step()
            with torch.no_grad():
                preds = (self.model(X_all.to(device)) > 0.5).float()
                acc = (preds.cpu() == y_all).float().mean().item()
            history.append(acc)
        return history


# ---------------------------------------------------------------------------
# Meta-learning utilities
# ---------------------------------------------------------------------------


class _LinearModel(torch.nn.Module):
    """Simple linear model used by the tests.

    The project contains numerous sophisticated architectures but for unit
    tests we only require a minimal model.  It exposes a binary classification
    head which allows us to demonstrate the benefits of meta-learning.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return self.fc(x).squeeze(1)


def steps_to(history: Sequence[float], thr: float = 0.9) -> int:
    """Return the number of optimisation steps required to reach ``thr``.

    The helper is primarily used in the training scripts to report the
    adaptation speed when a model is initialised from meta‑learned weights.
    It was previously exposed as ``_steps_to`` and is kept backwards
    compatible via an alias below so existing tests continue to function.
    """

    for i, a in enumerate(history, 1):
        if a >= thr:
            return i
    return len(history) + 1


# Backwards compatibility for older imports
_steps_to = steps_to


def meta_train_maml(
    tasks: Sequence[Tuple[TensorDataset, TensorDataset]],
    build_model: Callable[[], torch.nn.Module],
    inner_lr: float = 0.1,
    meta_lr: float = 0.01,
    inner_steps: int = 1,
    epochs: int = 5,
    state_dict: dict | None = None,
) -> dict:
    """Meta-train ``build_model`` using a first-order MAML update.

    ``state_dict`` can be provided to continue training from a previous
    meta-learned initialisation which is handy for curriculum based
    strategies where later stages refine earlier weights.
    """

    loss_fn = torch.nn.BCEWithLogitsLoss()
    meta_model = build_model()
    if state_dict:
        meta_model.load_state_dict(state_dict)

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


def meta_train_reptile(
    tasks: Sequence[Tuple[TensorDataset, TensorDataset]],
    build_model: Callable[[], torch.nn.Module],
    inner_lr: float = 0.1,
    meta_lr: float = 0.1,
    inner_steps: int = 1,
    epochs: int = 5,
    state_dict: dict | None = None,
) -> dict:
    """Meta-train ``build_model`` using the Reptile algorithm."""

    loss_fn = torch.nn.BCEWithLogitsLoss()
    meta_model = build_model()
    if state_dict:
        meta_model.load_state_dict(state_dict)

    for _ in range(epochs):
        for train_ds, _ in tasks:
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
            for meta_p, p in zip(meta_model.parameters(), model.parameters()):
                meta_p.data += meta_lr * (p.data - meta_p.data)
    return meta_model.state_dict()


def fine_tune_model(
    state_dict: dict,
    dataset: TensorDataset,
    build_model: Callable[[], torch.nn.Module],
    steps: int = 5,
    lr: float = 0.1,
) -> Tuple[dict, List[float]]:
    """Fine-tune a model initialised with ``state_dict`` on ``dataset``.

    Returns the new state dict together with a list of accuracy values after
    each optimisation step.  The helper mirrors :func:`analysis.meta_learning`
    but lives in the ``models`` package so that training scripts can import it
    without depending on the heavier analysis module.
    """

    model = build_model()
    model.load_state_dict(state_dict)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
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


__all__ = [
    "MetaLearner",
    "_LinearModel",
    "meta_train_maml",
    "meta_train_reptile",
    "fine_tune_model",
    "steps_to",
]

