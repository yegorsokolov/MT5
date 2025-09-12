import torch
from torch import nn


class Time2Vec(nn.Module):
    """Time2Vec encoding following [Kazemi2019]."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model < 2:
            raise ValueError("d_model must be at least 2 for Time2Vec")
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(d_model - 1))
        self.b = nn.Parameter(torch.randn(d_model - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1)
        linear = self.w0 * t + self.b0
        periodic = torch.sin(t * self.w + self.b)
        return torch.cat([linear, periodic], dim=-1)


class TimeEncoding(nn.Module):
    """Relative time encoding based on timestamp deltas."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.t2v = Time2Vec(d_model)

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        rel = times - times[..., :1]
        return self.t2v(rel)
