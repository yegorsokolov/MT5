import torch
from torch.nn.utils import clip_grad_norm_

def test_gradient_clipping():
    model = torch.nn.Linear(10, 1)
    # Create inputs and targets that produce large gradients
    inputs = torch.ones(5, 10)
    targets = torch.ones(5, 1) * 10
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(model(inputs), targets)
    loss.backward()
    # Compute norm before clipping
    total_norm = torch.sqrt(
        sum(p.grad.data.norm() ** 2 for p in model.parameters() if p.grad is not None)
    )
    assert total_norm > 1.0
    clip_grad_norm_(model.parameters(), 1.0)
    clipped_norm = torch.sqrt(
        sum(p.grad.data.norm() ** 2 for p in model.parameters() if p.grad is not None)
    )
    assert clipped_norm <= 1.0 + 1e-6

