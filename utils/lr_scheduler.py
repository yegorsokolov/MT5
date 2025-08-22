import torch
from torch.optim import AdamW, RMSprop
from typing import List

class LookaheadAdamW(AdamW):
    """AdamW optimizer with a Lookahead mechanism and variance-based LR adjustment.

    The optimizer tracks gradient variance and reduces the learning rate when
    variance increases while slightly increasing it when variance drops.  A
    Lookahead step averages fast weights with slow weights every ``k`` steps.
    """

    def __init__(self, params, lr: float = 1e-3, k: int = 5, alpha: float = 0.5,
                 beta: float = 0.98, min_lr: float = 1e-5, max_lr: float = 1e-2, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.min_lr = min_lr
        self.max_lr = max_lr
        self._step = 0
        self._avg_var = None
        self.slow_params: List[torch.Tensor] = [
            p.clone().detach()
            for group in self.param_groups for p in group["params"]
        ]

    def step(self, closure=None):  # type: ignore[override]
        grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grads.append(p.grad.detach().flatten())
        if grads:
            flat = torch.cat(grads)
            var = flat.var().item()
            if self._avg_var is None:
                self._avg_var = var
            if var > self._avg_var:
                scale = 0.5
            else:
                scale = 1.05
            for group in self.param_groups:
                lr = group["lr"] * scale
                lr = max(self.min_lr, min(self.max_lr, lr))
                group["lr"] = lr
            self._avg_var = self.beta * self._avg_var + (1 - self.beta) * var
        loss = super().step(closure)
        self._step += 1
        if self._step % self.k == 0:
            idx = 0
            for group in self.param_groups:
                for p in group["params"]:
                    slow = self.slow_params[idx]
                    slow.add_(self.alpha * (p.data - slow))
                    p.data.copy_(slow)
                    idx += 1
        return loss

    def get_lr(self) -> float:
        return float(self.param_groups[0]["lr"])


class VarianceRMSprop(RMSprop):
    """RMSprop optimizer that adapts the LR based on gradient variance."""

    def __init__(self, params, lr: float = 1e-3, beta: float = 0.98,
                 min_lr: float = 1e-5, max_lr: float = 1e-2, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.beta = beta
        self.min_lr = min_lr
        self.max_lr = max_lr
        self._avg_var = None

    def step(self, closure=None):  # type: ignore[override]
        grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grads.append(p.grad.detach().flatten())
        if grads:
            flat = torch.cat(grads)
            var = flat.var().item()
            if self._avg_var is None:
                self._avg_var = var
            if var > self._avg_var:
                scale = 0.5
            else:
                scale = 1.05
            for group in self.param_groups:
                lr = group["lr"] * scale
                lr = max(self.min_lr, min(self.max_lr, lr))
                group["lr"] = lr
            self._avg_var = self.beta * self._avg_var + (1 - self.beta) * var
        return super().step(closure)

    def get_lr(self) -> float:
        return float(self.param_groups[0]["lr"])
