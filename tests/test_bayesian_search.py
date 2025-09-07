"""Tests for the :mod:`tuning.bayesian_search` module."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tuning.bayesian_search import run_search


def _make_dataset() -> tuple[TensorDataset, TensorDataset]:
    """Return a tiny synthetic binary classification dataset."""
    torch.manual_seed(0)
    X = torch.randn(80, 4)
    y = (X.sum(dim=1) > 0).long()
    X_train, X_val = X[:60], X[60:]
    y_train, y_val = y[:60], y[60:]
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    return train_ds, val_ds


def _train_model(cfg: dict, trial: optuna.trial.Trial) -> float:
    train_ds, val_ds = _make_dataset()
    input_dim = train_ds.tensors[0].shape[1]
    hidden = 8
    layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
    for _ in range(int(cfg["max_depth"]) - 1):
        layers.append(nn.Linear(hidden, hidden))
        layers.append(nn.ReLU())
    if cfg["dropout"] > 0:
        layers.append(nn.Dropout(cfg["dropout"]))
    layers.append(nn.Linear(hidden, 2))
    model = nn.Sequential(*layers)

    opt = torch.optim.SGD(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()
    batch_size = int(cfg["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_acc = 0.0
    epochs_no_improve = 0
    for epoch in range(20):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += len(yb)
        acc = correct / max(1, total)
        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        if acc > best_acc:
            best_acc = acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= 3:
            break
    return best_acc


def test_search_logs_parameters(tmp_path: Path) -> None:
    base_cfg: dict = {}
    best = run_search(_train_model, base_cfg, n_trials=4, store_dir=tmp_path)
    # ensure parameters recorded
    tuned_files = list(tmp_path.glob("tuned_*.json"))
    assert tuned_files, "tuned parameters not saved"
    with open(tuned_files[0]) as fh:
        saved = json.load(fh)
    assert saved == best
    # Best configuration should achieve better than random accuracy
    final_score = _train_model({**base_cfg, **best}, optuna.trial.FixedTrial(best))
    assert final_score >= 0.5
