from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def distill_teacher_student(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    data_loader: DataLoader,
    epochs: int = 1,
    lr: float = 1e-3,
) -> torch.nn.Module:
    """Distill knowledge from ``teacher`` into ``student``.

    Parameters
    ----------
    teacher:
        Trained model providing soft targets.
    student:
        Smaller model to be trained to mimic the teacher.
    data_loader:
        Loader yielding input batches. Labels, if provided, are ignored.
    epochs:
        Number of distillation epochs.
    lr:
        Learning rate for the student's optimizer.

    Returns
    -------
    torch.nn.Module
        The trained student model.
    """
    device = next(student.parameters()).device
    teacher.eval()
    student.train()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for _ in range(max(1, epochs)):
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                xb = batch[0]
            else:
                xb = batch
            xb = xb.to(device)
            with torch.no_grad():
                teacher_out = teacher(xb)
            student_out = student(xb)
            loss = loss_fn(student_out, teacher_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return student
