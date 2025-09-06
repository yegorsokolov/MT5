from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def distill_teacher_student(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    data_loader: DataLoader,
    epochs: int = 1,
    lr: float = 1e-3,
    temperature: float = 1.0,
) -> torch.nn.Module:
    """Distill knowledge from ``teacher`` into ``student``.

    The routine is intentionally lightweight – it avoids any heavy training
    utilities so it can run in constrained test environments.  When the model
    outputs contain more than one dimension, a temperature scaled KL divergence
    is used, otherwise mean squared error is applied.  This allows the helper to
    work for both classification (logits) and regression style models.

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
    temperature:
        Softmax temperature for classification distillation. ``1.0`` disables
        temperature scaling.

    Returns
    -------
    torch.nn.Module
        The trained student model.
    """

    device = next(student.parameters()).device
    teacher.eval()
    student.train()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    kl_div = torch.nn.KLDivLoss(reduction="batchmean")
    mse = torch.nn.MSELoss()

    for _ in range(max(1, epochs)):
        for batch in data_loader:
            xb = batch[0] if isinstance(batch, (list, tuple)) else batch
            xb = xb.to(device)
            with torch.no_grad():
                teacher_out = teacher(xb)
            student_out = student(xb)

            if teacher_out.ndim > 1 and teacher_out.size(-1) > 1:
                t_logits = teacher_out / temperature
                s_logits = student_out / temperature
                loss = kl_div(
                    torch.log_softmax(s_logits, dim=-1),
                    torch.softmax(t_logits, dim=-1),
                ) * (temperature**2)
            else:
                loss = mse(student_out, teacher_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return student
