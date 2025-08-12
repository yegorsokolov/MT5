import importlib.util
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

spec = importlib.util.spec_from_file_location(
    "distillation", Path(__file__).resolve().parents[1] / "models" / "distillation.py"
)
distill_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(distill_mod)  # type: ignore
distill_teacher_student = distill_mod.distill_teacher_student


def _train_teacher(train_loader, model, epochs=50):
    loss_fn = torch.nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        for xb, yb in train_loader:
            optim.zero_grad()
            preds = model(xb).squeeze()
            loss = loss_fn(preds, yb)
            loss.backward()
            optim.step()
    return model


def _evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb).squeeze()
            pred_labels = (preds > 0.5).float()
            correct += (pred_labels == yb).sum().item()
            total += yb.size(0)
    return correct / total if total else 0.0


def _count_params(model):
    return sum(p.numel() for p in model.parameters())


def test_distillation_student_near_teacher_accuracy():
    torch.manual_seed(0)
    X = torch.randn(256, 2)
    y = (X.sum(dim=1) > 0).float()
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=32)

    teacher = torch.nn.Sequential(
        torch.nn.Linear(2, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
        torch.nn.Sigmoid(),
    )
    _train_teacher(train_loader, teacher, epochs=50)
    teacher_acc = _evaluate(teacher, test_loader)

    student = torch.nn.Sequential(
        torch.nn.Linear(2, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 1),
        torch.nn.Sigmoid(),
    )
    distill_teacher_student(teacher, student, train_loader, epochs=50, lr=0.01)
    student_acc = _evaluate(student, test_loader)

    assert _count_params(student) < _count_params(teacher)
    assert student_acc >= teacher_acc - 0.1
